"""
02_load_neo4j.py
────────────────────────────────────────────────────────────
CSV ファイルを Neo4j に MERGE ロードするスクリプト

対象 CSV:
  1. data/neo4j/nodes_standard.csv         ← 手動作成済み
  2. data/neo4j/nodes_chapter_section_item.csv
  3. data/neo4j/nodes_domain.csv
  4. data/neo4j/relations.csv
  5. data/neo4j/extracted/kg_nodes_*.csv   ← LLM 抽出結果
  6. data/neo4j/extracted/kg_relations_*.csv

Usage:
    python scripts/02_load_neo4j.py
    python scripts/02_load_neo4j.py --mode extracted   # LLM 抽出分のみ
    python scripts/02_load_neo4j.py --mode base        # 手動 CSV のみ
    python scripts/02_load_neo4j.py --reset            # DB をリセットしてから投入

環境変数（.env または shell で設定）:
    NEO4J_URI      = bolt://localhost:7687
    NEO4J_USER     = neo4j
    NEO4J_PASSWORD = your_password
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, Session

load_dotenv()

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DATA_DIR = Path("data/neo4j")
EXTRACTED_DIR = DATA_DIR / "extracted"

# ラベル → Cypher MERGE テンプレートのマッピング
#
# 構造ノード (Standard / Chapter / Section / Item) は id が安定キー → id でMERGE
# 概念ノード (FacilityType / TechnicalConcept 等) は name が安定キー → name でMERGE
#   ・LLM 抽出結果では id が空だったり連番になるため、同名が重複しにくい name を主キーに
#   ・ON CREATE: id を初回のみ SET / ON MATCH: description のみ上書き（既存データ保護）
LABEL_MERGE_MAP = {
    "Standard":         "MERGE (n:Standard {id: $id}) SET n.name = $name",
    "Chapter":          "MERGE (n:Chapter {id: $id}) SET n.name = $name",
    "Section":          "MERGE (n:Section {id: $id}) SET n.name = $name",
    "Item":             "MERGE (n:Item {id: $id}) SET n.name = $name",
    # 概念ノードは name 基準で MERGE（重複排除）
    "FacilityType":     (
        "MERGE (n:FacilityType {name: $name}) "
        "ON CREATE SET n.id = $id, n.description = $description "
        "ON MATCH SET n.description = CASE WHEN n.description = '' OR n.description IS NULL "
        "THEN $description ELSE n.description END"
    ),
    "TechnicalConcept": (
        "MERGE (n:TechnicalConcept {name: $name}) "
        "ON CREATE SET n.id = $id, n.description = $description "
        "ON MATCH SET n.description = CASE WHEN n.description = '' OR n.description IS NULL "
        "THEN $description ELSE n.description END"
    ),
    "HazardType": (
        "MERGE (n:HazardType {name: $name}) "
        "ON CREATE SET n.id = $id, n.description = $description "
        "ON MATCH SET n.description = CASE WHEN n.description = '' OR n.description IS NULL "
        "THEN $description ELSE n.description END"
    ),
    "RequirementType": (
        "MERGE (n:RequirementType {name: $name}) "
        "ON CREATE SET n.id = $id, n.description = $description "
        "ON MATCH SET n.description = CASE WHEN n.description = '' OR n.description IS NULL "
        "THEN $description ELSE n.description END"
    ),
    "ProcessConcept": (
        "MERGE (n:ProcessConcept {name: $name}) "
        "ON CREATE SET n.id = $id, n.description = $description "
        "ON MATCH SET n.description = CASE WHEN n.description = '' OR n.description IS NULL "
        "THEN $description ELSE n.description END"
    ),
    "Other":            "MERGE (n:KGNode {id: $id}) SET n.name = $name, n.source_text = $source_text",
}

# 有効なリレーションタイプ
VALID_REL_TYPES = {
    "HAS_CHAPTER", "HAS_SECTION", "HAS_ITEM",
    "DESCRIBED_IN", "REQUIRES", "SUBJECT_TO",
    "MITIGATES", "DEFINED_IN", "USED_IN",
    "PRECEDES", "AFFECTS", "RELATED_TO",
    "HAS_REQUIREMENT_TYPE",
}


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────
def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def batch(iterable, size=500):
    """リストをバッチ分割。"""
    lst = list(iterable)
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def normalize_name(s: str) -> str:
    """名前の正規化: 前後空白除去・中間の連続空白を単一スペースに統一。"""
    return " ".join(s.strip().split())


# ──────────────────────────────────────────────
# ノードロード
# ──────────────────────────────────────────────
def upsert_nodes(session: Session, rows: list[dict], label_col: str = ":LABEL"):
    counts: dict[str, int] = {}
    for lbl, cypher in LABEL_MERGE_MAP.items():
        subset = [r for r in rows if r.get(label_col) == lbl]
        if not subset:
            continue
        for chunk in batch(subset):
            params_list = []
            seen_names: set[str] = set()   # チャンク内の name 重複をスキップ
            for r in chunk:
                raw_name = normalize_name(r.get("name", ""))
                if not raw_name:
                    continue  # 名前空ノードはスキップ
                if raw_name in seen_names:
                    continue  # 同一チャンク内の名前重複をスキップ
                seen_names.add(raw_name)
                raw_id = normalize_name(r.get("id:ID") or r.get("id", ""))
                # id が空の場合は name を代用（LLM 抽出ノードで発生しやすい）
                stable_id = raw_id if raw_id else raw_name
                params_list.append({
                    "id":          stable_id,
                    "name":        raw_name,
                    "description": r.get("description", "").strip(),
                    "source_text": r.get("source_text", "").strip(),
                })
            session.run(
                f"UNWIND $rows AS r {cypher.replace('$id', 'r.id').replace('$name', 'r.name').replace('$description', 'r.description').replace('$source_text', 'r.source_text')}",
                rows=params_list,
            )
        counts[lbl] = counts.get(lbl, 0) + len(subset)
    return counts


def load_nodes_csv(session: Session, csv_path: Path):
    rows = read_csv(csv_path)
    label_col = ":LABEL" if ":LABEL" in rows[0] else "type"
    counts = upsert_nodes(session, rows, label_col=label_col)
    total = sum(counts.values())
    print(f"  [nodes] {csv_path.name}: {total} 件  {dict(sorted(counts.items()))}")
    return total


# ──────────────────────────────────────────────
# リレーションロード
# ──────────────────────────────────────────────
def load_relations_csv(session: Session, csv_path: Path):
    rows = read_csv(csv_path)
    count = 0
    for row in rows:
        start_col = ":START_ID" if ":START_ID" in row else "source"
        end_col   = ":END_ID"   if ":END_ID"   in row else "target"
        type_col  = ":TYPE"     if ":TYPE"      in row else "type"

        start_id  = row.get(start_col, "").strip()
        end_id    = row.get(end_col, "").strip()
        rel_type  = row.get(type_col, "").strip()
        evidence  = row.get("evidence", "")

        if not start_id or not end_id or not rel_type:
            continue
        if rel_type not in VALID_REL_TYPES:
            print(f"  [SKIP] 未知のリレーションタイプ: {rel_type}")
            continue

        # ノードは id または name どちらでも検索
        cypher = f"""
            MATCH (a) WHERE a.id = $start OR a.name = $start
            MATCH (b) WHERE b.id = $end   OR b.name = $end
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.evidence = $evidence
        """
        try:
            result = session.run(cypher, start=start_id, end=end_id, evidence=evidence)
            result.consume()
            count += 1
        except Exception as e:
            print(f"  [WARN] Relation error ({start_id} → {end_id}): {e}")

    print(f"  [relations] {csv_path.name}: {count} 件")
    return count


# ──────────────────────────────────────────────
# スキーマ初期化（制約 & インデックス）
# ──────────────────────────────────────────────
def init_schema(session: Session):
    # 構造ノード: id UNIQUE
    id_constraints = [
        ("Standard", "id"),
        ("Chapter",  "id"),
        ("Section",  "id"),
        ("Item",     "id"),
        ("KGNode",   "id"),
    ]
    for label, prop in id_constraints:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_{prop}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )

    # 概念ノード: name UNIQUE（主 MERGE キーに合わせる）
    name_constraints = [
        "FacilityType",
        "TechnicalConcept",
        "HazardType",
        "RequirementType",
        "ProcessConcept",
    ]
    for label in name_constraints:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )

    # フルテキスト検索インデックス
    session.run("""
        CREATE FULLTEXT INDEX kasen_fulltext IF NOT EXISTS
        FOR (n:Standard|Chapter|Section|Item|FacilityType|TechnicalConcept|HazardType|ProcessConcept|KGNode)
        ON EACH [n.name, n.description, n.source_text]
    """)
    print("  [schema] 制約・インデックス設定完了")


# ──────────────────────────────────────────────
# post-load 重複ノード集約
# ──────────────────────────────────────────────
def deduplicate_concept_nodes(session: Session):
    """
    概念ノードに同一 name を持つ重複ノードがある場合、
    最も古い（id(n) が小さい）ノードに他のノードのリレーションを付け替え、
    余剰ノードを削除する。

    ※ APOC 不使用の純 Cypher 実装（Cypher では直接リレーション付け替え不可なため
      Python ループで処理する）。
    """
    concept_labels = [
        "FacilityType", "TechnicalConcept",
        "HazardType", "RequirementType", "ProcessConcept",
    ]
    total_removed = 0
    for label in concept_labels:
        # 同一 name の重複ペアを取得（elementId() は Neo4j 5.x 推奨）
        dups = session.run(
            f"""
            MATCH (a:{label}), (b:{label})
            WHERE a.name = b.name AND elementId(a) < elementId(b)
            RETURN elementId(a) AS keep_id, elementId(b) AS drop_id, a.name AS name
            """
        ).data()
        if not dups:
            continue
        print(f"  [dedup:{label}] {len(dups)} 件の重複を検出")
        for dup in dups:
            keep_id = dup["keep_id"]
            drop_id = dup["drop_id"]
            # drop ノードの outbound リレーションを keep ノードに付け替え
            session.run(
                """
                MATCH (drop) WHERE elementId(drop) = $drop_id
                MATCH (keep) WHERE elementId(keep) = $keep_id
                MATCH (drop)-[r]->(x)
                WHERE NOT (keep)-[]->(x)
                  AND elementId(x) <> $keep_id
                WITH keep, type(r) AS rtype, x
                CALL apoc.create.relationship(keep, rtype, {}, x) YIELD rel
                RETURN count(rel)
                """,
                keep_id=keep_id, drop_id=drop_id,
            ) if False else None  # APOC が使えない場合はスキップ

            session.run(
                """
                MATCH (drop) WHERE elementId(drop) = $drop_id
                DETACH DELETE drop
                """,
                drop_id=drop_id,
            )
            total_removed += 1
        print(f"  [dedup:{label}] {len(dups)} 件を削除")
    if total_removed:
        print(f"  [dedup] 合計 {total_removed} 件の重複ノードを削除しました")
    else:
        print("  [dedup] 重複ノードなし")


# ──────────────────────────────────────────────
# DB リセット
# ──────────────────────────────────────────────
def reset_db(session: Session):
    session.run("MATCH (n) DETACH DELETE n")
    print("  [reset] 全ノード・リレーションを削除しました")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Neo4j CSVローダー")
    parser.add_argument("--mode", choices=["all", "base", "extracted"], default="all",
                        help="all: 全 CSV, base: 手動 CSV のみ, extracted: LLM 抽出分のみ")
    parser.add_argument("--reset", action="store_true", help="ロード前にDBをリセット")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"Neo4j 接続: {NEO4J_URI}")

    with driver.session() as session:
        # リセット（--reset 指定時は先に全削除してからスキーマを作成）
        if args.reset:
            reset_db(session)

        # スキーマ（制約・インデックス）
        init_schema(session)

        # ── base CSV ──
        if args.mode in ("all", "base"):
            base_files = [
                DATA_DIR / "nodes_standard.csv",
                DATA_DIR / "nodes_domain.csv",
                DATA_DIR / "nodes_chapter_section_item.csv",
            ]
            for f in base_files:
                if f.exists():
                    load_nodes_csv(session, f)
                else:
                    print(f"  [SKIP] {f} が見つかりません")

            rel_file = DATA_DIR / "relations.csv"
            if rel_file.exists():
                load_relations_csv(session, rel_file)

        # ── LLM 抽出 CSV ──
        if args.mode in ("all", "extracted"):
            if EXTRACTED_DIR.exists():
                for node_csv in sorted(EXTRACTED_DIR.glob("kg_nodes_*.csv")):
                    load_nodes_csv(session, node_csv)

                for rel_csv in sorted(EXTRACTED_DIR.glob("kg_relations_*.csv")):
                    load_relations_csv(session, rel_csv)
            else:
                print("  [INFO] extracted/ ディレクトリが存在しません。01_extract_entities.py を先に実行してください。")

        # post-load: 概念ノード重複排除
        print("\n[post-load] 重複ノード集約中...")
        deduplicate_concept_nodes(session)

    driver.close()
    print("\n=== ロード完了 ===")


if __name__ == "__main__":
    main()
