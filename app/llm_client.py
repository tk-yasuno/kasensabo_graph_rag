"""
app/llm_client.py  ―  LLM クライアント（OpenAI / ローカル vLLM / Ollama 対応）

GPU 16GB 環境での利用を想定:
  - OpenAI API (開発・評価フェーズ)
  - vLLM サーバー (4bit 量子化)
  - Ollama + GPT-OSS Swallow 20B RL v0.1 (東京科学大学×産総研 日本語強化推論型)

Ollama 使用時:
  GPT-OSS 系の特殊チャンネルトークン形式に対応するため
  OpenAI 互換エンドポイントではなく /api/generate (raw モード) を使用。
  チャットテンプレートを手動で組み立てることで確実に動作させる。
"""
from __future__ import annotations

import os
import httpx
from openai import OpenAI

from app.config import settings

# ──────────────────────────────────────────────
# Ollama 判定ユーティリティ
# ──────────────────────────────────────────────
def _is_ollama() -> bool:
    return "11434" in settings.LLM_BASE_URL or "ollama" in settings.LLM_BASE_URL.lower()


def _ollama_base() -> str:
    """http://localhost:11434 の形式に正規化"""
    base = settings.LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


# ──────────────────────────────────────────────
# OpenAI クライアント（OpenAI 公式 / vLLM 用）
# ──────────────────────────────────────────────
def _make_client() -> OpenAI:
    kwargs: dict = {"api_key": settings.OPENAI_API_KEY}
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return OpenAI(**kwargs)


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# ──────────────────────────────────────────────
# Ollama ネイティブ API 呼び出し
# GPT-OSS 系は特殊チャンネルトークンを使うため
# OpenAI 互換 API 経由では応答が空になる。
# raw=True + /api/generate で手動テンプレートを使う。
# ──────────────────────────────────────────────
def _ollama_chat(system: str, user: str, max_tokens: int = 2048) -> str:
    """
    Ollama /api/generate (raw モード) を使って GPT-OSS Swallow に問い合わせる。
    GPT-OSS チャンネルトークン形式:
      <|start|>role<|channel|>ch<|message|>content<|end|>
    """
    prompt = (
        f"<|start|>system<|message|>{system}<|end|>\n"
        f"<|start|>user<|message|>{user}<|end|>\n"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    url = f"{_ollama_base()}/api/generate"
    resp = httpx.post(
        url,
        json={
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": settings.LLM_TEMP,
                "num_predict": max_tokens,
                "num_ctx": 8192,          # コンテキストウィンドウ拡張（デフォルト 2048 → 8192）
                "repeat_penalty": 1.2,    # 繰り返しループ抑制（Case A / C 共通）
                "stop": ["<|end|>", "<|start|>"],
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

# ──────────────────────────────────────────────
# プロンプトテンプレート
# ──────────────────────────────────────────────
SYSTEM_PROMPT_GRAPHRAG = """\
あなたは「河川砂防技術基準」の専門家アシスタントです。
以下の【知識グラフコンテキスト】は、Neo4j から取得した技術基準の構造化情報です。

回答のガイドライン:
1. 技術基準の「考え方」「必須」「標準」「推奨」「例示」の区別を明示する
2. 施設名・章節名など固有の専門用語は正確に使用する
3. 根拠となる基準名・章・節を必ず示す（例: 維持管理編（河川編） 第6章 第2節）
4. 推測や基準外の内容を含める場合は明示する
5. 実務者が判断できるレベルの具体的な回答を心がける
6. 質問の用語を勝手に言い換えたり再解釈したりしないこと
   （例: 「浸食」を「浸水」に読み替えるといった変換は禁止）
7. コンテキストに情報が不足している場合は、技術基準の一般的知識で補い、
   その旨を「※基準コンテキスト外」と明記すること
"""

CONTEXT_TEMPLATE = """\
【知識グラフコンテキスト】
{graph_context}

【質問】
{question}
"""


def build_context_text(graph_records: list[dict], max_chars: int = 2000) -> str:
    """
    グラフ検索結果を LLM に渡せるテキストに変換。
    max_chars: コンテキスト文字数の上限（超えた分は省略）。
    num_ctx=8192 の場合、入力に約 4000 字割り当てれば出力に 2000+ トークン確保できる。
    """
    if not graph_records:
        return "(関連ノードが見つかりませんでした)"

    lines: list[str] = []
    seen: set[str] = set()

    for r in graph_records:
        node_name  = r.get("node_name") or r.get("facility") or r.get("concept") or ""
        node_label = r.get("node_label", "")
        rel_type   = r.get("rel_type", "")
        nbr_name   = r.get("neighbor_name") or ""
        nbr_label  = r.get("neighbor_label", "")

        # ノード自体
        if node_name and node_name not in seen:
            desc = r.get("facility_desc") or r.get("concept_desc") or ""
            lines.append(f"- [{node_label}] {node_name}" + (f"  ← {desc}" if desc else ""))
            seen.add(node_name)

            # リスト系フィールド
            for key, label in [
                ("hazards",               "対象ハザード"),
                ("required_concepts",     "必要な技術概念"),
                ("described_in",          "関連基準章節"),
                ("mitigates_hazards",     "軽減するハザード"),
                ("affected_facilities",   "影響を受ける施設"),
                ("mitigating_facilities", "対策施設"),
                ("items",                 "項目"),
            ]:
                vals = [v for v in (r.get(key) or []) if v]
                if vals:
                    lines.append(f"  {label}: {', '.join(vals)}")

        # リレーションと隣接ノード
        if rel_type and nbr_name and nbr_name not in seen:
            lines.append(f"  → [{rel_type}] → [{nbr_label}] {nbr_name}")

    result = "\n".join(lines)
    if max_chars and len(result) > max_chars:
        result = result[:max_chars] + "\n...(コンテキスト長超過のため省略)"
    return result


# ──────────────────────────────────────────────
# LLM 呼び出し
# ──────────────────────────────────────────────
def answer_with_context(question: str, graph_records: list[dict]) -> str:
    """
    グラフ検索結果をコンテキストとして LLM に与え、回答を生成する。
    Ollama 使用時は /api/generate (raw) でチャンネルトークンを手動処理。
    """
    ctx_text = build_context_text(graph_records)
    user_msg = CONTEXT_TEMPLATE.format(
        graph_context=ctx_text,
        question=question,
    )

    if _is_ollama():
        return _ollama_chat(SYSTEM_PROMPT_GRAPHRAG, user_msg, max_tokens=2048)

    client = get_client()
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GRAPHRAG},
            {"role": "user",   "content": user_msg},
        ],
        temperature=settings.LLM_TEMP,
    )
    return resp.choices[0].message.content


def answer_plain(question: str) -> str:
    """
    コンテキストなし（ケース A: プレーン LLM）で回答する。
    Ollama 使用時は /api/generate (raw) でチャンネルトークンを手動処理。
    """
    system = "あなたは河川・ダム・砂防の専門家です。質問に正確に日本語で答えてください。"

    if _is_ollama():
        return _ollama_chat(system, question, max_tokens=2048)

    client = get_client()
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=settings.LLM_TEMP,
    )
    return resp.choices[0].message.content
