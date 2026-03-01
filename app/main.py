"""
app/main.py  ―  FastAPI エントリーポイント

起動方法:
    uvicorn app.main:app --reload --port 8080

エンドポイント一覧:
    GET  /              ヘルスチェック
    POST /query         GraphRAG で回答（ケース C）
    POST /query/plain   プレーン LLM 回答（ケース A）
    POST /compare       ケース A / C を同じ質問で比較
    GET  /graph/facility/{name}    施設の関連情報を取得
    GET  /graph/hazard/{name}      ハザードの関連施設を取得
    GET  /graph/standard/{std_id}  基準の章節階層を取得
    GET  /graph/maintenance        維持管理技術概念一覧
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.graph_rag import run_case_a, run_case_c
from app.neo4j_client import KGQ, close_driver, session

# ──────────────────────────────────────────────
# アプリ初期化
# ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "河川砂防技術基準（調査・計画・設計・維持管理編）の"
        "知識グラフ × LLM によるグラフ RAG API"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# スキーマ
# ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., example="砂防堰堤の点検はどのように行いますか？")


class CompareRequest(BaseModel):
    question: str = Field(..., example="砂防堰堤と床固工の設計条件の違いを教えてください。")


class QueryResponse(BaseModel):
    case: str
    question: str
    answer: str
    graph_hits: list[dict]
    context_preview: str


class CompareResponse(BaseModel):
    question: str
    case_a: dict
    case_c: dict


# ──────────────────────────────────────────────
# ライフサイクル
# ──────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown_event():
    close_driver()


# ──────────────────────────────────────────────
# エンドポイント
# ──────────────────────────────────────────────
@app.get("/", tags=["health"])
def health():
    return {
        "status": "ok",
        "title":   settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "model":   settings.LLM_MODEL,
    }


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query_graphrag(req: QueryRequest):
    """
    **ケース C: GraphRAG**

    Neo4j 知識グラフで関連ノードを検索し、その情報を LLM のコンテキストとして
    渡して回答を生成します。
    """
    try:
        result = run_case_c(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/plain", response_model=QueryResponse, tags=["rag"])
def query_plain(req: QueryRequest):
    """
    **ケース A: プレーン LLM**

    知識グラフを使わず、LLM のみで回答します（ベースライン比較用）。
    """
    try:
        result = run_case_a(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse, tags=["rag"])
def compare_cases(req: CompareRequest):
    """
    **ケース A / C の比較**

    同じ質問に対してプレーン LLM と GraphRAG の両方で回答し、並べて返します。
    """
    try:
        result_a = run_case_a(req.question)
        result_c = run_case_c(req.question)
        return {
            "question": req.question,
            "case_a": result_a,
            "case_c": result_c,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/facility/{name}", tags=["graph"])
def get_facility_context(name: str):
    """
    施設名から関連ハザード・技術概念・基準章節を取得。

    例: `/graph/facility/砂防堰堤`
    """
    try:
        with session() as s:
            records = s.execute_read(KGQ.facility_context, facility_name=name)
        if not records:
            raise HTTPException(status_code=404, detail=f"施設 '{name}' は見つかりませんでした")
        return {"facility": name, "records": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/hazard/{name}", tags=["graph"])
def get_hazard_context(name: str):
    """
    ハザード名から影響を受ける施設・対策施設を取得。

    例: `/graph/hazard/土石流`
    """
    try:
        with session() as s:
            records = s.execute_read(KGQ.hazard_facility_map, hazard_name=name)
        if not records:
            raise HTTPException(status_code=404, detail=f"ハザード '{name}' は見つかりませんでした")
        return {"hazard": name, "records": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/standard/{std_id}", tags=["graph"])
def get_standard_hierarchy(std_id: str):
    """
    基準 ID から章・節・項の階層構造を取得。

    例: `/graph/standard/STD_IJIKANRI_KASEN`
    """
    try:
        with session() as s:
            records = s.execute_read(KGQ.standard_hierarchy, standard_id=std_id)
        if not records:
            raise HTTPException(status_code=404, detail=f"基準 '{std_id}' は見つかりませんでした")
        return {"standard_id": std_id, "hierarchy": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/maintenance", tags=["graph"])
def get_maintenance_concepts():
    """
    維持管理プロセスに関連する技術概念と適用施設の一覧を取得。
    """
    try:
        with session() as s:
            records = s.execute_read(KGQ.maintenance_cycle_query)
        return {"maintenance_concepts": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
