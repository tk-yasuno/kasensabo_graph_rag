// ============================================================
// Neo4j 知識グラフ 初期スキーマ構築スクリプト
// 河川砂防技術基準 GraphRAG MVP
// 実行手順: Neo4j Browser / Cypher Shell で上から順に実行
// ============================================================

// ────────────────────────────────────────────────
// STEP 1: 制約（Constraint）設定
// ────────────────────────────────────────────────
CREATE CONSTRAINT std_id IF NOT EXISTS
  FOR (s:Standard) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT chapter_id IF NOT EXISTS
  FOR (c:Chapter) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT section_id IF NOT EXISTS
  FOR (s:Section) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT item_id IF NOT EXISTS
  FOR (i:Item) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT facility_id IF NOT EXISTS
  FOR (f:FacilityType) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT tech_id IF NOT EXISTS
  FOR (t:TechnicalConcept) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT hazard_id IF NOT EXISTS
  FOR (h:HazardType) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT req_id IF NOT EXISTS
  FOR (r:RequirementType) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT process_id IF NOT EXISTS
  FOR (p:ProcessConcept) REQUIRE p.id IS UNIQUE;

// ────────────────────────────────────────────────
// STEP 2: フルテキスト検索インデックス
//   RAG クエリの高速化（name / description を対象）
// ────────────────────────────────────────────────
CREATE FULLTEXT INDEX kasen_fulltext IF NOT EXISTS
  FOR (n:Standard|Chapter|Section|Item|FacilityType|TechnicalConcept|HazardType|ProcessConcept)
  ON EACH [n.name, n.description];

// ────────────────────────────────────────────────
// STEP 3: CSV ロード（nodes_standard.csv）
// ────────────────────────────────────────────────
// ※ Neo4j の import フォルダ（%NEO4J_HOME%/import/）に CSV を配置してから実行
LOAD CSV WITH HEADERS FROM 'file:///nodes_standard.csv' AS row
MERGE (s:Standard {id: row.`id:ID`})
SET s.name = row.name;

// ────────────────────────────────────────────────
// STEP 4: CSV ロード（nodes_chapter_section_item.csv）
// ────────────────────────────────────────────────
// Chapter
LOAD CSV WITH HEADERS FROM 'file:///nodes_chapter_section_item.csv' AS row
WITH row WHERE row.`:LABEL` = 'Chapter'
MERGE (c:Chapter {id: row.`id:ID`})
SET c.name = row.name, c.parent_id = row.`parent_id`;

// Section
LOAD CSV WITH HEADERS FROM 'file:///nodes_chapter_section_item.csv' AS row
WITH row WHERE row.`:LABEL` = 'Section'
MERGE (s:Section {id: row.`id:ID`})
SET s.name = row.name, s.parent_id = row.`parent_id`;

// Item
LOAD CSV WITH HEADERS FROM 'file:///nodes_chapter_section_item.csv' AS row
WITH row WHERE row.`:LABEL` = 'Item'
MERGE (i:Item {id: row.`id:ID`})
SET i.name = row.name, i.parent_id = row.`parent_id`;

// ────────────────────────────────────────────────
// STEP 5: CSV ロード（nodes_domain.csv）
// ────────────────────────────────────────────────
// FacilityType
LOAD CSV WITH HEADERS FROM 'file:///nodes_domain.csv' AS row
WITH row WHERE row.`:LABEL` = 'FacilityType'
MERGE (f:FacilityType {id: row.`id:ID`})
SET f.name = row.name, f.description = row.description;

// TechnicalConcept
LOAD CSV WITH HEADERS FROM 'file:///nodes_domain.csv' AS row
WITH row WHERE row.`:LABEL` = 'TechnicalConcept'
MERGE (t:TechnicalConcept {id: row.`id:ID`})
SET t.name = row.name, t.description = row.description;

// HazardType
LOAD CSV WITH HEADERS FROM 'file:///nodes_domain.csv' AS row
WITH row WHERE row.`:LABEL` = 'HazardType'
MERGE (h:HazardType {id: row.`id:ID`})
SET h.name = row.name, h.description = row.description;

// RequirementType
LOAD CSV WITH HEADERS FROM 'file:///nodes_domain.csv' AS row
WITH row WHERE row.`:LABEL` = 'RequirementType'
MERGE (r:RequirementType {id: row.`id:ID`})
SET r.name = row.name, r.description = row.description;

// ProcessConcept
LOAD CSV WITH HEADERS FROM 'file:///nodes_domain.csv' AS row
WITH row WHERE row.`:LABEL` = 'ProcessConcept'
MERGE (p:ProcessConcept {id: row.`id:ID`})
SET p.name = row.name, p.description = row.description;

// ────────────────────────────────────────────────
// STEP 6: リレーションのロード（relations.csv）
// APOC がある場合は apoc.create.relationship を使用
// APOC がない場合は以下の代替クエリを使用
// ────────────────────────────────────────────────

// APOC あり（推奨）:
// LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
// MATCH (a {id: row.`:START_ID`})
// MATCH (b {id: row.`:END_ID`})
// CALL apoc.create.relationship(a, row.`:TYPE`, {}, b) YIELD rel
// RETURN count(rel);

// APOC なし（手動パターン展開）:
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
MATCH (a {id: row.`:START_ID`})
MATCH (b {id: row.`:END_ID`})
WITH a, b, row
CALL {
  WITH a, b, row
  // HAS_CHAPTER
  WITH a, b, row WHERE row.`:TYPE` = 'HAS_CHAPTER'
  MERGE (a)-[:HAS_CHAPTER]->(b)
}
CALL {
  WITH a, b, row
  // HAS_SECTION
  WITH a, b, row WHERE row.`:TYPE` = 'HAS_SECTION'
  MERGE (a)-[:HAS_SECTION]->(b)
}
CALL {
  WITH a, b, row
  // HAS_ITEM
  WITH a, b, row WHERE row.`:TYPE` = 'HAS_ITEM'
  MERGE (a)-[:HAS_ITEM]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'DESCRIBED_IN'
  MERGE (a)-[:DESCRIBED_IN]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'REQUIRES'
  MERGE (a)-[:REQUIRES]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'SUBJECT_TO'
  MERGE (a)-[:SUBJECT_TO]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'MITIGATES'
  MERGE (a)-[:MITIGATES]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'DEFINED_IN'
  MERGE (a)-[:DEFINED_IN]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'USED_IN'
  MERGE (a)-[:USED_IN]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'PRECEDES'
  MERGE (a)-[:PRECEDES]->(b)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.`:TYPE` = 'AFFECTS'
  MERGE (a)-[:AFFECTS]->(b)
}
RETURN count(*);

// ────────────────────────────────────────────────
// STEP 7: 検証クエリ（ロード後に実行して確認）
// ────────────────────────────────────────────────

// ノード数の確認
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC;

// 砂防堰堤の関連ノードを確認
MATCH (f:FacilityType {id: 'FAC_SABO_DAM'})
OPTIONAL MATCH (f)-[r]->(x)
RETURN f.name AS facility, type(r) AS rel_type, x.name AS target
ORDER BY rel_type;

// ダム維持管理の階層構造を確認
MATCH (s:Standard {id: 'STD_IJIKANRI_DAM'})-[:HAS_CHAPTER]->(ch:Chapter)
OPTIONAL MATCH (ch)-[:HAS_SECTION]->(sec:Section)
RETURN s.name AS standard, ch.name AS chapter, sec.name AS section
ORDER BY ch.name, sec.name;

// 洪水に関わる施設一覧
MATCH (h:HazardType {id: 'HZ_FLOOD'})-[:AFFECTS]->(f:FacilityType)
RETURN h.name AS hazard, f.name AS facility;

// ────────────────────────────────────────────────
// STEP 8: GraphRAG 用クエリテンプレート
// ────────────────────────────────────────────────

// Q1: キーワードによる関連ノード検索（ベーシック RAG クエリ）
// パラメータ: $keyword
//
// MATCH (n)
// WHERE n.name CONTAINS $keyword OR n.description CONTAINS $keyword
// WITH n LIMIT 30
// OPTIONAL MATCH (n)-[r]->(x)
// RETURN n, type(r) AS rel, x
// LIMIT 60;

// Q2: 施設 × ハザード × 基準の三角形クエリ
// パラメータ: $facility_name, $hazard_name
//
// MATCH (f:FacilityType)-[:SUBJECT_TO]->(h:HazardType)
// MATCH (f)-[:DESCRIBED_IN]->(ch)
// WHERE f.name CONTAINS $facility_name AND h.name CONTAINS $hazard_name
// RETURN f.name AS facility, h.name AS hazard, ch.name AS standard_chapter
// LIMIT 20;

// Q3: 維持管理プロセスに必要な技術概念と施設
// パラメータ: $process (例: 維持管理)
//
// MATCH (tc:TechnicalConcept)-[:USED_IN]->(pc:ProcessConcept)
// WHERE pc.name CONTAINS $process
// OPTIONAL MATCH (f:FacilityType)-[:REQUIRES]->(tc)
// RETURN tc.name AS concept, collect(DISTINCT f.name) AS facilities
// ORDER BY tc.name;

// Q4: 施設から関連する技術基準の条文（章・節・項）を取得
// パラメータ: $facility_id
//
// MATCH (f:FacilityType {id: $facility_id})-[:DESCRIBED_IN]->(ch)
// OPTIONAL MATCH (ch)-[:HAS_SECTION]->(sec)
// OPTIONAL MATCH (sec)-[:HAS_ITEM]->(item)
// RETURN ch.name AS chapter, sec.name AS section, collect(item.name) AS items;
