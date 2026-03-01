"""
app/graph_rag.py  ―  GraphRAG オーケストレーター

RAG フロー:
  1. 質問 → キーワード抽出（施設名・ハザード名を検出）
  2. Neo4j に対して複数の Cypher クエリを並行実行
  3. 取得結果 → コンテキスト整形
  4. LLM に渡して最終回答を生成
"""
from __future__ import annotations

import math
import re

from app.config import settings
from app.llm_client import answer_plain, answer_with_context, build_context_text
from app.neo4j_client import KGQ, session

# ──────────────────────────────────────────────
# キーワード抽出（簡易版: 施設名・ハザード名を正規表現で検出）
# ──────────────────────────────────────────────
FACILITY_KEYWORDS = [
    "堤防", "高規格堤防", "護岸", "水制", "床止め", "堰", "樋門", "水門",
    "排水機場", "ダム", "コンクリートダム", "フィルダム",
    "砂防堰堤", "床固工", "山腹工", "渓流保全工", "土石流導流工",
    "地すべり防止施設", "急傾斜地崩壊防止施設", "雪崩対策施設",
]

HAZARD_KEYWORDS = [
    "洪水", "土石流", "地すべり", "急傾斜地崩壊", "雪崩", "津波", "高潮", "浸水",
]

MAINTENANCE_KEYWORDS = [
    "点検", "維持管理", "長寿命化", "健全度", "予防保全", "修繕",
    "定期点検", "臨時点検", "詳細点検",
]

# graph_hits がこの閾値を下回ったとき GRAPH_TOP_K × 2 で再検索する
GRAPH_LOW_HIT_THRESHOLD = 25


def extract_keywords(question: str) -> dict:
    """質問テキストから主要キーワードを抽出する。"""
    found_facilities = [f for f in FACILITY_KEYWORDS if f in question]
    found_hazards    = [h for h in HAZARD_KEYWORDS    if h in question]
    found_maint      = [m for m in MAINTENANCE_KEYWORDS if m in question]

    # フォールバック: 先頭 30 字をクエリとして使用
    generic_kw = question[:40].replace("\n", " ").strip()

    return {
        "facilities": found_facilities,
        "hazards":    found_hazards,
        "maintenance": found_maint,
        "generic":    generic_kw,
    }


# ──────────────────────────────────────────────
# 関連スコア計算（再ランキング用）
# ──────────────────────────────────────────────
def _score_record(rec: dict, question: str) -> float:
    """
    レコードの関連スコアを算出する。
    - フルテキスト検索由来: Neo4j の score 値 × 10 を使用（他指標と桁を揃える）
    - その他: 質問テキストと施設/ハザード/維持管理キーワードの一致数を返す。
    """
    if rec.get("score") is not None:
        return float(rec["score"]) * 10.0

    # テキスト系フィールドを全連結
    text = " ".join(
        str(v) for v in rec.values()
        if isinstance(v, str) and v
    )
    # リスト型フィールドも展開
    for v in rec.values():
        if isinstance(v, list):
            text += " " + " ".join(str(x) for x in v if x)

    # 質問に含まれるキーワードが rec テキストに出現する数を加算
    hits = sum(
        1
        for kw in (FACILITY_KEYWORDS + HAZARD_KEYWORDS + MAINTENANCE_KEYWORDS)
        if kw in question and kw in text
    )
    return float(hits)


# ──────────────────────────────────────────────
# グラフ検索
# ──────────────────────────────────────────────
def _run_queries(kw: dict, top_k: int, *, extra_kw: str | None = None) -> list[dict]:
    """
    Neo4j に対して一連のクエリを実行し、生レコードリストを返す（重複含む）。
    extra_kw: 追加フォールバック検索キーワード（リトライ時に使用）。
    """
    records: list[dict] = []
    with session() as s:
        # 1. フルテキスト検索（汎用）
        records.extend(s.execute_read(KGQ.keyword_search, keyword=kw["generic"], top_k=top_k))

        # 2. 施設コンテキスト
        for fac in kw["facilities"][:3]:
            records.extend(s.execute_read(KGQ.facility_context, facility_name=fac))

        # 3. ハザードマップ
        for haz in kw["hazards"][:2]:
            records.extend(s.execute_read(KGQ.hazard_facility_map, hazard_name=haz))

        # 4. 維持管理クエリ
        if kw["maintenance"]:
            records.extend(s.execute_read(KGQ.maintenance_cycle_query))

        # 5. 施設比較（「〇〇と△△の違い」パターン）
        if len(kw["facilities"]) >= 2:
            records.extend(
                s.execute_read(
                    KGQ.compare_facilities,
                    name_a=kw["facilities"][0],
                    name_b=kw["facilities"][1],
                )
            )

        # 6. リトライ時の追加広域検索（Section/Chapter 名前部分一致）
        if extra_kw:
            records.extend(s.execute_read(KGQ.broad_section_search, keyword=extra_kw, top_k=top_k))

    return records


def _deduplicate(records: list[dict]) -> list[dict]:
    """(node_id, rel_type, neighbor_id) をキーに重複排除する。"""
    seen: set = set()
    unique: list[dict] = []
    for r in records:
        key = (r.get("node_id"), r.get("rel_type"), r.get("neighbor_id"))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def retrieve_graph_context(question: str) -> list[dict]:
    """
    質問内容に応じた Cypher クエリを実行し、関連レコードを返す。
    graph_hits が GRAPH_LOW_HIT_THRESHOLD を下回る場合は
    GRAPH_TOP_K × 2 で自動リトライする。
    """
    kw = extract_keywords(question)
    top_k = settings.GRAPH_TOP_K

    records = _run_queries(kw, top_k)

    # 重複排除（node_id ベース）
    unique = _deduplicate(records)

    # ── 適応的リトライ: hits が閾値未満なら TOP_K × 2 ＋ 広域検索 ──
    if len(unique) < GRAPH_LOW_HIT_THRESHOLD:
        extra_kw = question[:20].replace("\n", " ").strip()  # 質問先頭20字で广域検索
        retry_records = _run_queries(kw, top_k * 2, extra_kw=extra_kw)
        # 元のレコードを含めて再度重複排除
        unique = _deduplicate(records + retry_records)

    if not unique:
        return unique

    # ── スコア付きリストを作成・降順ソート ──
    scored = sorted(
        [(r, _score_record(r, question)) for r in unique],
        key=lambda x: x[1],
        reverse=True,
    )

    # ノイズ除去 step1: score > 0 のレコードがあれば score=0 をノイズとして除外
    has_positive = scored[0][1] > 0
    if has_positive:
        scored = [(r, s) for r, s in scored if s > 0]

    # ノイズ除去 step2: 残ったレコードの上位 GRAPH_RERANK_RATIO (80%) のみ残す
    keep_n = max(1, math.ceil(len(scored) * settings.GRAPH_RERANK_RATIO))
    return [r for r, _ in scored[:keep_n]]


# ──────────────────────────────────────────────
# ケース A / B / C の回答生成
# ──────────────────────────────────────────────
def run_case_a(question: str) -> dict:
    """
    ケース A: プレーン LLM（知識グラフ・RAG なし）
    """
    answer = answer_plain(question)
    return {
        "case": "A_plain_llm",
        "question": question,
        "answer": answer,
        "graph_hits": [],
        "context_preview": "",
    }


def run_case_c(question: str) -> dict:
    """
    ケース C: GraphRAG（Neo4j 知識グラフ + LLM）
    """
    graph_hits = retrieve_graph_context(question)
    ctx_text   = build_context_text(graph_hits)
    answer     = answer_with_context(question, graph_hits)

    return {
        "case": "C_graph_rag",
        "question": question,
        "answer": answer,
        "graph_hits": graph_hits,
        "context_preview": ctx_text[:800],  # API 返却用に先頭 800 字
    }
