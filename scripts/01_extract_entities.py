"""
01_extract_entities.py
────────────────────────────────────────────────────────────
河川砂防技術基準 Markdown → エンティティ・リレーション抽出
LLM (OpenAI) を使って JSON で出力し、Neo4j ロード用 CSV を生成

Usage:
    python scripts/01_extract_entities.py \
        --input data/kasen-dam-sabo_Train_set/05_training_ijikanri_kasen_2025.md \
        --output data/neo4j/extracted \
        --model gpt-4o-mini \
        --chunk_size 1200
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 環境変数を使用

# ──────────────────────────────────────────────
# 1. チャンク化
# ──────────────────────────────────────────────
def chunk_by_paragraph(text: str, max_len: int = 1200) -> list[dict]:
    """
    Markdown テキストを段落・見出し単位に分割する。
    Returns: list of {"chunk_text": str, "header": str}
    """
    chunks = []
    lines = text.split("\n")
    current_lines: list[str] = []
    current_header = "本文"
    length = 0

    for line in lines:
        # 見出し（# / ## / ### など）でチャンクを区切る
        if re.match(r"^#{1,4}\s", line):
            if current_lines:
                chunks.append({
                    "header": current_header,
                    "chunk_text": "\n".join(current_lines).strip(),
                })
                current_lines = []
                length = 0
            current_header = line.strip("# ").strip()

        current_lines.append(line)
        length += len(line)

        # max_len を超えたら強制的に区切る
        if length >= max_len:
            chunks.append({
                "header": current_header,
                "chunk_text": "\n".join(current_lines).strip(),
            })
            current_lines = []
            length = 0

    if current_lines:
        chunks.append({
            "header": current_header,
            "chunk_text": "\n".join(current_lines).strip(),
        })

    return [c for c in chunks if c["chunk_text"]]


# ──────────────────────────────────────────────
# 2. LLM によるエンティティ・リレーション抽出
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
あなたは「河川砂防技術基準」の情報抽出専門 AI です。
与えられたテキストから、以下の指示に従って JSON のみを返してください。

抽出する型:
- FacilityType  : 河川・砂防施設（堤防、砂防堰堤、ダム など）
- TechnicalConcept : 技術概念・手法（水文解析、長寿命化計画、点検 など）
- HazardType    : 自然災害・外力（洪水、土石流、地すべり など）
- RequirementType  : 適用区分（必須、標準、推奨、考え方、例示）
- Other         : 上記に分類できないが重要なエンティティ

抽出するリレーションの種類:
- DESCRIBED_IN  : 施設が基準に説明されている
- REQUIRES      : 施設・概念が手法・点検を必要とする
- SUBJECT_TO    : 施設がハザードに晒される
- MITIGATES     : 施設がハザードを軽減する
- DEFINED_IN    : 概念が基準で定義される
- RELATED_TO    : その他の関連

出力形式（JSON のみ。説明文不要）:
{
  "entities": [
    {"name": "...", "type": "...", "source_text": "...（原文の根拠箇所）"}
  ],
  "relations": [
    {"source": "...", "target": "...", "type": "...", "evidence": "...（原文抜粋）"}
  ]
}
"""

def extract_from_chunk(chunk: dict, model: str, retry: int = 3) -> dict:
    """LLM を呼び出し、エンティティと関係の JSON を返す。"""
    user_msg = f"[見出し: {chunk['header']}]\n\n{chunk['chunk_text']}"
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] API error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return {"entities": [], "relations": []}


# ──────────────────────────────────────────────
# 3. 重複排除 & ID 生成
# ──────────────────────────────────────────────
def normalize_id(name: str, etype: str) -> str:
    """エンティティを一意の ID 文字列（ASCII 互換）に変換。"""
    import hashlib
    key = f"{etype}_{name}"
    return f"{etype[:3].upper()}_{hashlib.md5(key.encode()).hexdigest()[:8]}"


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for e in entities:
        key = (e["name"], e["type"])
        if key not in seen:
            seen[key] = e
    return list(seen.values())


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result = []
    for r in relations:
        key = (r["source"], r["target"], r["type"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


# ──────────────────────────────────────────────
# 4. CSV 書き出し
# ──────────────────────────────────────────────
def write_nodes_csv(nodes: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id:ID", "name", ":LABEL", "source_text"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for n in nodes:
            writer.writerow({
                "id:ID": normalize_id(n["name"], n["type"]),
                "name": n["name"],
                ":LABEL": n["type"],
                "source_text": n.get("source_text", ""),
            })
    print(f"  → ノード CSV: {out_path} ({len(nodes)} 件)")


def write_relations_csv(relations: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[":START_ID", ":END_ID", ":TYPE", "evidence"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in relations:
            # source / target を ID に変換するため名前のまま保存（後工程で JOIN）
            writer.writerow({
                ":START_ID": r.get("source", ""),
                ":END_ID": r.get("target", ""),
                ":TYPE": r.get("type", ""),
                "evidence": r.get("evidence", ""),
            })
    print(f"  → リレーション CSV: {out_path} ({len(relations)} 件)")


# ──────────────────────────────────────────────
# 5. メイン処理
# ──────────────────────────────────────────────
def process_document(
    input_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    chunk_size: int = 1200,
    delay: float = 0.5,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    stem = input_path.stem

    print(f"\n=== 処理開始: {input_path.name} ===")
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_by_paragraph(text, max_len=chunk_size)
    print(f"  チャンク数: {len(chunks)}")

    all_entities: list[dict] = []
    all_relations: list[dict] = []

    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] header={chunk['header'][:40]!r}")
        result = extract_from_chunk(chunk, model)
        all_entities.extend(result.get("entities", []))
        all_relations.extend(result.get("relations", []))
        time.sleep(delay)

    all_entities = deduplicate_entities(all_entities)
    all_relations = deduplicate_relations(all_relations)

    write_nodes_csv(all_entities, output_dir / f"kg_nodes_{stem}.csv")
    write_relations_csv(all_relations, output_dir / f"kg_relations_{stem}.csv")

    print(f"=== 完了: エンティティ {len(all_entities)} 件 / リレーション {len(all_relations)} 件 ===\n")


def main():
    parser = argparse.ArgumentParser(description="技術基準 → KG エンティティ抽出")
    parser.add_argument("--input", required=True, help="入力 Markdown ファイルパス")
    parser.add_argument("--output", default="data/neo4j/extracted", help="出力ディレクトリ")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI モデル名")
    parser.add_argument("--chunk_size", type=int, default=1200, help="チャンクの最大文字数")
    parser.add_argument("--delay", type=float, default=0.5, help="API 呼び出し間の待機秒数")
    args = parser.parse_args()

    process_document(
        input_path=args.input,
        output_dir=args.output,
        model=args.model,
        chunk_size=args.chunk_size,
        delay=args.delay,
    )


if __name__ == "__main__":
    # デフォルト実行例（全 MD ファイルを処理）
    import sys
    if len(sys.argv) == 1:
        train_dir = Path("data/kasen-dam-sabo_Train_set")
        for md_file in sorted(train_dir.glob("*.md")):
            process_document(
                input_path=str(md_file),
                output_dir="data/neo4j/extracted",
                model="gpt-4o-mini",
                chunk_size=1200,
                delay=0.5,
            )
    else:
        main()
