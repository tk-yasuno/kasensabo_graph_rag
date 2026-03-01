"""
app/config.py  ―  環境変数・設定値の一元管理
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# プロジェクトルートの .env を読み込む
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    # ── Neo4j ──────────────────────────────────
    NEO4J_URI:      str = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER:     str = os.getenv("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # ── OpenAI / ローカル LLM ──────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # ローカル LLM を使う場合は base_url を変更してください
    # 例: http://localhost:11434/v1  (Ollama)
    #     http://localhost:8000/v1   (vLLM)
    LLM_BASE_URL:   str = os.getenv("LLM_BASE_URL",  "")   # 空文字 = OpenAI 公式
    LLM_MODEL:      str = os.getenv("LLM_MODEL",     "gpt-4o-mini")

    # ── GraphRAG ──────────────────────────────
    GRAPH_TOP_K:         int   = int(os.getenv("GRAPH_TOP_K",      "20"))   # 各サブクエリの Neo4j 検索幅
    GRAPH_RERANK_RATIO:  float = float(os.getenv("GRAPH_RERANK_RATIO", "0.8"))  # ノイズ除去閾値: スコア上位 80% を使用
    LLM_TEMP:       float = float(os.getenv("LLM_TEMP", "0.2"))

    # ── アプリ ──────────────────────────────────
    APP_TITLE:      str = "河川砂防 GraphRAG API"
    APP_VERSION:    str = "0.1.0"
    DEBUG:          bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
