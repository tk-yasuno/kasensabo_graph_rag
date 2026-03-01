"""
app/neo4j_client.py  ―  Neo4j 接続 & Cypher クエリ集
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

from app.config import settings

# ──────────────────────────────────────────────
# ドライバシングルトン
# ──────────────────────────────────────────────
_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


@contextmanager
def session():
    drv = get_driver()
    with drv.session() as s:
        yield s


# ──────────────────────────────────────────────
# Cypher クエリ群
# ──────────────────────────────────────────────
class KnowledgeGraphQueries:

    @staticmethod
    def keyword_search(tx: ManagedTransaction, keyword: str, top_k: int = 20) -> list[dict]:
        """
        フリーワード検索: 名前・説明に keyword を含むノードと周辺を取得。
        """
        cypher = """
        CALL db.index.fulltext.queryNodes('kasen_fulltext', $kw)
        YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT $top_k
        OPTIONAL MATCH (node)-[r]->(neighbor)
        RETURN
          node.id    AS node_id,
          node.name  AS node_name,
          labels(node)[0] AS node_label,
          type(r)    AS rel_type,
          neighbor.id   AS neighbor_id,
          neighbor.name AS neighbor_name,
          labels(neighbor)[0] AS neighbor_label,
          score
        ORDER BY score DESC
        """
        result = tx.run(cypher, kw=keyword, top_k=top_k)
        return [r.data() for r in result]

    @staticmethod
    def facility_context(tx: ManagedTransaction, facility_name: str) -> list[dict]:
        """
        施設名から: 関連ハザード・技術概念・基準章節を取得。
        """
        cypher = """
        MATCH (f:FacilityType)
        WHERE f.name CONTAINS $name
        OPTIONAL MATCH (f)-[:SUBJECT_TO]->(h:HazardType)
        OPTIONAL MATCH (f)-[:REQUIRES]->(tc:TechnicalConcept)
        OPTIONAL MATCH (f)-[:DESCRIBED_IN]->(ch)
        OPTIONAL MATCH (f)-[:MITIGATES]->(mh:HazardType)
        RETURN
          f.name AS facility,
          f.description AS facility_desc,
          collect(DISTINCT h.name)  AS hazards,
          collect(DISTINCT tc.name) AS required_concepts,
          collect(DISTINCT ch.name) AS described_in,
          collect(DISTINCT mh.name) AS mitigates_hazards
        LIMIT $top_k
        """
        result = tx.run(cypher, name=facility_name, top_k=10)
        return [r.data() for r in result]

    @staticmethod
    def standard_hierarchy(tx: ManagedTransaction, standard_id: str) -> list[dict]:
        """
        基準 ID → 章 → 節 → 項の階層展開。
        """
        cypher = """
        MATCH (s:Standard {id: $sid})-[:HAS_CHAPTER]->(ch:Chapter)
        OPTIONAL MATCH (ch)-[:HAS_SECTION]->(sec:Section)
        OPTIONAL MATCH (sec)-[:HAS_ITEM]->(item:Item)
        RETURN
          s.name   AS standard,
          ch.name  AS chapter,
          sec.name AS section,
          collect(item.name) AS items
        ORDER BY ch.name, sec.name
        """
        result = tx.run(cypher, sid=standard_id)
        return [r.data() for r in result]

    @staticmethod
    def hazard_facility_map(tx: ManagedTransaction, hazard_name: str) -> list[dict]:
        """
        ハザード → 影響を受ける施設・緩和施設を取得。
        """
        cypher = """
        MATCH (h:HazardType)
        WHERE h.name CONTAINS $name
        OPTIONAL MATCH (h)-[:AFFECTS]->(f:FacilityType)
        OPTIONAL MATCH (f2:FacilityType)-[:MITIGATES]->(h)
        RETURN
          h.name AS hazard,
          collect(DISTINCT f.name)  AS affected_facilities,
          collect(DISTINCT f2.name) AS mitigating_facilities
        """
        result = tx.run(cypher, name=hazard_name)
        return [r.data() for r in result]

    @staticmethod
    def maintenance_cycle_query(tx: ManagedTransaction) -> list[dict]:
        """
        維持管理プロセスの技術概念一覧。
        """
        cypher = """
        MATCH (tc:TechnicalConcept)-[:USED_IN]->(pc:ProcessConcept {name: "維持管理"})
        OPTIONAL MATCH (f:FacilityType)-[:REQUIRES]->(tc)
        RETURN
          tc.name AS concept,
          tc.description AS concept_desc,
          collect(DISTINCT f.name) AS applied_facilities
        ORDER BY tc.name
        """
        result = tx.run(cypher)
        return [r.data() for r in result]

    @staticmethod
    def compare_facilities(tx: ManagedTransaction, name_a: str, name_b: str) -> list[dict]:
        """
        2つの施設の設計基準・ハザード・技術概念を比較取得。
        """
        cypher = """
        MATCH (f:FacilityType)
        WHERE f.name CONTAINS $a OR f.name CONTAINS $b
        OPTIONAL MATCH (f)-[:SUBJECT_TO]->(h:HazardType)
        OPTIONAL MATCH (f)-[:REQUIRES]->(tc:TechnicalConcept)
        OPTIONAL MATCH (f)-[:DESCRIBED_IN]->(ch)
        RETURN
          f.name AS facility,
          collect(DISTINCT h.name)  AS hazards,
          collect(DISTINCT tc.name) AS concepts,
          collect(DISTINCT ch.name) AS chapters
        """
        result = tx.run(cypher, a=name_a, b=name_b)
        return [r.data() for r in result]

    @staticmethod
    def broad_section_search(tx: ManagedTransaction, keyword: str, top_k: int = 40) -> list[dict]:
        """
        graph_hits 不足時の広域フォールバック検索。
        Chapter / Section ノードを名前部分一致で検索し、関連 Item と TechnicalConcept を返す。
        keyword は質問先頭 20 字等の短縮テキストを渡す。
        """
        cypher = """
        MATCH (n)
        WHERE (n:Chapter OR n:Section OR n:TechnicalConcept)
          AND (n.name CONTAINS $kw OR n.description CONTAINS $kw)
        OPTIONAL MATCH (n)-[r]->(neighbor)
        RETURN
          n.id    AS node_id,
          n.name  AS node_name,
          labels(n)[0] AS node_label,
          type(r)       AS rel_type,
          neighbor.id   AS neighbor_id,
          neighbor.name AS neighbor_name,
          labels(neighbor)[0] AS neighbor_label,
          null AS score
        LIMIT $top_k
        """
        result = tx.run(cypher, kw=keyword, top_k=top_k)
        return [r.data() for r in result]


# デフォルトインスタンス
KGQ = KnowledgeGraphQueries()
