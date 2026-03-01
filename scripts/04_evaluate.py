"""
scripts/04_evaluate.py  ―  GraphRAG vs プレーン LLM 評価スクリプト

概要:
    test_questions_100.json の各質問を Case A（プレーン LLM）と
    Case C（GraphRAG）の両方に投げ、LLM-as-Judge で自動採点する。

出力:
    data/eval/results/results_<timestamp>.jsonl   各問の詳細結果
    data/eval/results/summary_<timestamp>.md      集計レポート

使い方:
    # 全100問（LLM-as-Judge あり）
    python scripts/04_evaluate.py

    # 指定範囲のみ
    python scripts/04_evaluate.py --start 1 --end 10

    # Judge なし（回答収集のみ）
    python scripts/04_evaluate.py --no-judge

    # 既存の結果ファイルを読み込んで Judge のみ再実行
    python scripts/04_evaluate.py --judge-only results/results_xxx.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Windows cp932 環境で絵文字・特殊文字を含む出力が失敗しないよう UTF-8 に固定
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class TeeLogger:
    """
    sys.stdout に調び、同時に UTF-8 ファイルへも書き出すラッパー。
    PowerShell の Tee-Object が cp932 で上書きする問題を回避する。
    """
    def __init__(self, filepath: Path) -> None:
        self._file = filepath.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str) -> int:
        self._file.write(msg)
        return self._stdout.write(msg)

    def flush(self) -> None:
        self._file.flush()
        self._stdout.flush()

    def close(self) -> None:
        self._file.close()

# ─── パス設定 ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

QUESTIONS_FILE = ROOT / "data" / "eval" / "test_questions_100.json"
RESULTS_DIR    = ROOT / "data" / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_BASE    = "http://localhost:8080"
TIMEOUT     = 300.0   # 秒（2モデル競合時のスワップロード時間を考慮して長めに設定）
RETRY_MAX   = 2
RETRY_WAIT  = 10.0

# ─── LLM-as-Judge 設定 ───────────────────────────────────
# RAG 実行モデル（GPT-OSS Swallow 20B）と Judge モデル（Qwen2.5:14B）を分離
# → プレーン LLM 評価で「自分の回答を自分で採点」するバイアスを回避
OLLAMA_BASE  = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("LLM_MODEL", "")
_OLLAMA_URL  = OLLAMA_BASE.replace("/v1", "").rstrip("/")
JUDGE_MODEL  = os.getenv("JUDGE_MODEL", "qwen2.5:14b")  # Judge 用は第三者モデル

JUDGE_SYSTEM = """あなたは河川砂防技術の専門家として、AI が生成した回答を採点します。
以下の採点基準に従い、必ず次の 2 行形式のみで返答してください。
他の文章れは一切不要です。

採点基準:
  3点: 技術的に正確で具体的、根拠となる基準名・章番号・技術概念が含まれる
  2点: 概ね正確だが、根拠・具体性がやや不足
  1点: 部分的に正しいが、重要な誤り・不足がある
  0点: 回答なし、または技術的に大きく誤っている

必ず次の 2 行形式のみで返答:
SCORE: <0、3の整数>
REASON: <50字以内の日本語での採点理由>"""

JUDGE_USER_TEMPLATE = """【質問】
{question}

【回答】
{answer}

SCORE: と REASON: の 2 行形式で採点してください。"""


# ─── ユーティリティ ──────────────────────────────────────
def _call_api(method: str, url: str, **kwargs) -> dict:
    """FastAPI エンドポイントを呼び出す（リトライあり）。"""
    client = httpx.Client(timeout=TIMEOUT)
    for attempt in range(RETRY_MAX + 1):
        try:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < RETRY_MAX:
                print(f"    ⚠ リトライ {attempt + 1}/{RETRY_MAX}: {e}")
                time.sleep(RETRY_WAIT)
            else:
                return {"error": str(e), "answer": "", "graph_hits": []}
    return {}


def _parse_judge_text(text: str) -> dict | None:
    """
    Judge モデルの出力を解析して {"score": N, "reason": "..."} を返す。
    優先順:
      1. SCORE: N / REASON: ... 行形式（プロンプトで要求する形式）
      2. {"score": N, ...} JSON 形式（フォールバック）
      3. テキスト中のスコア数字から拾う最終フォールバック
    """
    # 1. SCORE: N 形式
    sm = re.search(r'SCORE[:\uff1a]\s*([0-3])', text, re.IGNORECASE)
    if sm:
        rm = re.search(r'REASON[:\uff1a]\s*(.{1,100})', text, re.IGNORECASE)
        reason = rm.group(1).strip().rstrip('\n') if rm else ""
        return {"score": int(sm.group(1)), "reason": reason}

    # 2. JSON 形式フォールバック
    candidates = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if "score" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 3. 数字のみ最終フォールバック
    m = re.search(r'(?:score|スコア|得点|評価点|点)[:\uff1a\s]*([0-3])', text, re.IGNORECASE)
    if m:
        rm = re.search(r'(?:reason|理由)[:\uff1a\s](.{1,80})', text, re.IGNORECASE)
        reason = rm.group(1).strip() if rm else text[:50]
        return {"score": int(m.group(1)), "reason": f"[fb] {reason}"}

    return None


def _ollama_judge(question: str, answer: str) -> dict:
    """
    Qwen2.5:14B を Judge として /api/chat 経由で呼び出す。
    RAG 実行モデル（GPT-OSS Swallow）と分離することで自己採点バイアスを回避。
    """
    if not answer or not answer.strip():
        return {"score": 0, "reason": "回答なし"}

    retry_suffixes = [
        "",
        "\n\n必ず「SCORE: <数字>」「REASON: <理由>」の 2 行のみで回答してください。",
    ]

    text = ""
    for attempt, suffix in enumerate(retry_suffixes):
        user_text = JUDGE_USER_TEMPLATE.format(
            question=question,
            answer=answer[:1000],
        ) + suffix

        # Qwen2.5 は標準チャット形式（/api/chat）で呼び出す
        payload = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_text},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200, "num_ctx": 4096},
        }

        try:
            resp = httpx.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=300.0)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            result = _parse_judge_text(text)
            if result is not None:
                return result
            if attempt == 0:
                time.sleep(2.0)
        except Exception as e:
            if attempt == len(retry_suffixes) - 1:
                return {"score": -1, "reason": f"Judge エラー: {e}"}
            time.sleep(2.0)

    return {"score": -1, "reason": f"SCORE 抽出失敗（全リトライ消費）: {text[:60]}"}


# ─── メイン評価ロジック ──────────────────────────────────
def evaluate_question(q: dict, use_judge: bool) -> dict:
    """1問を評価し、結果辞書を返す。"""
    qid      = q["id"]
    question = q["question"]
    category = q.get("category", "")
    subcat   = q.get("subcategory", "")

    print(f"  [Q{qid:03d}] {category}/{subcat}")
    print(f"         {question[:60]}...")

    # Case A: プレーン LLM
    t0 = time.perf_counter()
    res_a = _call_api("POST", f"{API_BASE}/query/plain",
                      json={"question": question})
    time_a = time.perf_counter() - t0
    answer_a = res_a.get("answer", "")

    # Case C: GraphRAG
    t0 = time.perf_counter()
    res_c = _call_api("POST", f"{API_BASE}/query",
                      json={"question": question})
    time_c = time.perf_counter() - t0
    answer_c   = res_c.get("answer", "")
    graph_hits = len(res_c.get("graph_hits", []))

    print(f"         Case A: {len(answer_a)}字  Case C: {len(answer_c)}字  "
          f"graph_hits: {graph_hits}  A:{time_a:.1f}s  C:{time_c:.1f}s")

    # LLM-as-Judge
    judge_a: dict = {}
    judge_c: dict = {}
    if use_judge:
        judge_a = _ollama_judge(question, answer_a)
        judge_c = _ollama_judge(question, answer_c)
        score_a = judge_a.get("score", -1)
        score_c = judge_c.get("score", -1)
        print(f"         Judge  A:{score_a}点  C:{score_c}点")

    return {
        "id":           qid,
        "category":     category,
        "subcategory":  subcat,
        "question":     question,
        "case_a": {
            "answer":     answer_a,
            "length":     len(answer_a),
            "elapsed_s":  round(time_a, 2),
            "judge":      judge_a,
        },
        "case_c": {
            "answer":     answer_c,
            "length":     len(answer_c),
            "elapsed_s":  round(time_c, 2),
            "graph_hits": graph_hits,
            "judge":      judge_c,
        },
    }


def run_judge_only(results_path: Path) -> None:
    """既存の結果ファイルに Judge のみを適用して上書きする。"""
    records = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    updated = []
    for r in records:
        print(f"  Judge Q{r['id']:03d}...")
        r["case_a"]["judge"] = _ollama_judge(r["question"], r["case_a"]["answer"])
        r["case_c"]["judge"] = _ollama_judge(r["question"], r["case_c"]["answer"])
        updated.append(r)

    results_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n",
        encoding="utf-8",
    )
    print(f"Judge 結果を上書き保存: {results_path}")
    generate_summary(updated, results_path.with_suffix(".md"))


def generate_summary(records: list[dict], out_path: Path) -> None:
    """評価結果の集計レポートを Markdown で生成する。"""
    total = len(records)
    if total == 0:
        return

    has_judge = any(r["case_a"].get("judge") for r in records)

    # 基本統計
    avg_len_a = sum(r["case_a"]["length"] for r in records) / total
    avg_len_c = sum(r["case_c"]["length"] for r in records) / total
    avg_hits  = sum(r["case_c"]["graph_hits"] for r in records) / total
    avg_time_a = sum(r["case_a"]["elapsed_s"] for r in records) / total
    avg_time_c = sum(r["case_c"]["elapsed_s"] for r in records) / total

    # Judge スコア
    scores_a, scores_c = [], []
    if has_judge:
        for r in records:
            sa = r["case_a"]["judge"].get("score", -1)
            sc = r["case_c"]["judge"].get("score", -1)
            if sa >= 0:
                scores_a.append(sa)
            if sc >= 0:
                scores_c.append(sc)

    def score_dist(scores: list[int]) -> str:
        if not scores:
            return "N/A"
        dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in scores:
            dist[s] = dist.get(s, 0) + 1
        return "  ".join(f"{k}点:{v}問" for k, v in dist.items())

    # カテゴリ別集計
    from collections import defaultdict
    cat_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "sum_a": 0, "sum_c": 0, "n_judge": 0
    })
    for r in records:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        if has_judge:
            sa = r["case_a"]["judge"].get("score", -1)
            sc = r["case_c"]["judge"].get("score", -1)
            if sa >= 0 and sc >= 0:
                cat_stats[cat]["sum_a"] += sa
                cat_stats[cat]["sum_c"] += sc
                cat_stats[cat]["n_judge"] += 1

    # レポート生成
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# GraphRAG 評価レポート",
        f"",
        f"生成日時: {ts} / 評価問数: {total}問",
        f"",
        f"---",
        f"",
        f"## 全体サマリー",
        f"",
        f"| 指標 | Case A（プレーン LLM）| Case C（GraphRAG）|",
        f"|---|---|---|",
        f"| 平均回答文字数 | {avg_len_a:.0f} 字 | {avg_len_c:.0f} 字 |",
        f"| 平均応答時間 | {avg_time_a:.1f} 秒 | {avg_time_c:.1f} 秒 |",
        f"| 平均 graph_hits | — | {avg_hits:.1f} 件 |",
    ]

    if has_judge and scores_a and scores_c:
        avg_sa = sum(scores_a) / len(scores_a)
        avg_sc = sum(scores_c) / len(scores_c)
        lines += [
            f"| Judge 平均スコア（/3） | {avg_sa:.2f} | {avg_sc:.2f} |",
        ]

    lines += ["", "---", "", "## スコア分布（LLM-as-Judge）", ""]

    if has_judge:
        lines += [
            f"- **Case A**: {score_dist(scores_a)}  (有効 {len(scores_a)}/{total}問)",
            f"- **Case C**: {score_dist(scores_c)}  (有効 {len(scores_c)}/{total}問)",
        ]
    else:
        lines.append("Judge なし（`--no-judge` で実行）")

    lines += ["", "---", "", "## カテゴリ別平均スコア", ""]

    if has_judge:
        lines += [
            "| カテゴリ | 問数 | Case A avg | Case C avg | GraphRAG 効果 |",
            "|---|---|---|---|---|",
        ]
        for cat, st in sorted(cat_stats.items()):
            n = st["n_judge"]
            if n > 0:
                avg_a_cat = st["sum_a"] / n
                avg_c_cat = st["sum_c"] / n
                delta = avg_c_cat - avg_a_cat
                arrow = f"▲ +{delta:.2f}" if delta > 0 else (f"▼ {delta:.2f}" if delta < 0 else "→ 0.00")
                lines.append(
                    f"| {cat} | {st['total']}問 | {avg_a_cat:.2f} | {avg_c_cat:.2f} | {arrow} |"
                )
    else:
        for cat, st in sorted(cat_stats.items()):
            lines.append(f"- {cat}: {st['total']}問")

    # 問別詳細（Judge スコアがある場合は差分も表示）
    lines += ["", "---", "", "## 問別詳細", "", "| # | カテゴリ | 質問（先頭40字）| A長 | C長 | hits | A点 | C点 |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        q_short = r["question"][:40].replace("|", "｜")
        la = r["case_a"]["length"]
        lc = r["case_c"]["length"]
        hits = r["case_c"]["graph_hits"]
        sa_str = str(r["case_a"]["judge"].get("score", "—")) if r["case_a"].get("judge") else "—"
        sc_str = str(r["case_c"]["judge"].get("score", "—")) if r["case_c"].get("judge") else "—"
        lines.append(f"| {r['id']} | {r['category']} | {q_short} | {la} | {lc} | {hits} | {sa_str} | {sc_str} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📊 レポート保存: {out_path}")


# ─── エントリーポイント ──────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG 評価スクリプト")
    parser.add_argument("--start", type=int, default=1,    help="開始問番号（1始まり）")
    parser.add_argument("--end",   type=int, default=100,  help="終了問番号（含む）")
    parser.add_argument("--no-judge", action="store_true", help="LLM-as-Judge をスキップ")
    parser.add_argument("--judge-only", type=str, default="", help="既存 JSONL に judge のみ適用")
    parser.add_argument("--questions", type=str, default=str(QUESTIONS_FILE), help="質問 JSON ファイル")
    parser.add_argument("--output",    type=str, default="",  help="出力 JSONL ファイル（省略時は自動命名）")
    parser.add_argument("--sleep", type=float, default=1.0, help="問間のスリープ秒（デフォルト 1.0）")
    parser.add_argument("--log",   type=str, default="",    help="ログ出力先ファイル（UTF-8、要指定の場合）")
    args = parser.parse_args()

    # --log 指定時は TeeLogger でリダイレクト（Python から直接 UTF-8 書き出し）
    tee: TeeLogger | None = None
    if args.log:
        tee = TeeLogger(Path(args.log))
        sys.stdout = tee  # type: ignore[assignment]

    # Judge-only モード
    if args.judge_only:
        run_judge_only(Path(args.judge_only))
        return

    # 質問ロード
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in questions if args.start <= q["id"] <= args.end]
    total = len(questions)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl  = Path(args.output) if args.output else RESULTS_DIR / f"results_{ts}.jsonl"
    out_md     = out_jsonl.with_suffix(".md")

    use_judge = not args.no_judge

    print(f"{'=' * 60}")
    print(f"GraphRAG 評価 [{args.start}〜{args.end}問 / 計{total}問]")
    print(f"Judge: {'あり' if use_judge else 'なし'}")
    print(f"出力:  {out_jsonl}")
    print(f"{'=' * 60}")

    records: list[dict] = []
    fh = out_jsonl.open("w", encoding="utf-8")

    try:
        for i, q in enumerate(questions, 1):
            print(f"\n──── {i}/{total} ────")
            result = evaluate_question(q, use_judge)
            records.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            if i < total:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n\n⚠ 中断されました。途中経過を保存します...")
    finally:
        fh.close()

    print(f"\n✅ 評価完了 ({len(records)}/{total} 問)")
    print(f"   結果: {out_jsonl}")

    # サマリーレポート生成
    generate_summary(records, out_md)

    # ターミナル簡易サマリー
    if records:
        has_judge = any(r["case_a"].get("judge") for r in records)
        avg_len_a = sum(r["case_a"]["length"] for r in records) / len(records)
        avg_len_c = sum(r["case_c"]["length"] for r in records) / len(records)
        avg_hits  = sum(r["case_c"]["graph_hits"] for r in records) / len(records)
        print(f"\n--- 簡易サマリー ---")
        print(f"平均文字数  A: {avg_len_a:.0f}字  C: {avg_len_c:.0f}字")
        print(f"平均 graph_hits: {avg_hits:.1f}件")
        if has_judge:
            sa_list = [r["case_a"]["judge"].get("score", -1) for r in records if r["case_a"].get("judge")]
            sc_list = [r["case_c"]["judge"].get("score", -1) for r in records if r["case_c"].get("judge")]
            sa_valid = [s for s in sa_list if s >= 0]
            sc_valid = [s for s in sc_list if s >= 0]
            if sa_valid and sc_valid:
                print(f"Judge 平均  A: {sum(sa_valid)/len(sa_valid):.2f}/3  "
                      f"C: {sum(sc_valid)/len(sc_valid):.2f}/3")

    if tee is not None:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
