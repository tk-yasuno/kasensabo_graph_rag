"""
03b_generate_lora_qa_graph.py
────────────────────────────────────────────────────────────
Neo4j グラフリレーション駆動 LoRA 学習用 QA ペア生成

戦略:
  - 268 リレーション × 3問 = 804問
  - テスト100問と独立（文字 bigram Jaccard 類似度 < 0.5 でフィルタ）
  - Ollama（ローカル LLM）使用。OpenAI API は不要
  - 出力: Alpaca instruction 形式 JSONL

Usage:
    python scripts/03b_generate_lora_qa_graph.py
    python scripts/03b_generate_lora_qa_graph.py --model qwen2.5:14b --n_per_rel 3
    python scripts/03b_generate_lora_qa_graph.py --dry_run   # 文脈確認のみ
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
OLLAMA_BASE     = "http://localhost:11434"
DEFAULT_MODEL   = "hf.co/mmnga-o/GPT-OSS-Swallow-20B-RL-v0.1-gguf:Q4_K_M"
N_PER_REL       = 3          # リレーション1件あたりの生成問数
SIMILARITY_THRESH = 0.45     # テスト問題との最大 bigram Jaccard
REQUEST_TIMEOUT = 120        # seconds
RETRY           = 2
INTER_DELAY     = 0.5        # 秒（API 間隔）

NODE_CSVS   = [
    "data/neo4j/nodes_standard.csv",
    "data/neo4j/nodes_chapter_section_item.csv",
    "data/neo4j/nodes_domain.csv",
]
RELATION_CSV    = "data/neo4j/relations.csv"
TEST_QUESTIONS  = "data/eval/test_questions_100.json"
OUT_DIR         = Path("data/lora")

# ──────────────────────────────────────────────
# リレーション種別 → 日本語文脈テンプレート
# ──────────────────────────────────────────────
REL_TEMPLATE = {
    "HAS_CHAPTER":  "{src}は「{tgt}」を含む。",
    "HAS_SECTION":  "{src}は「{tgt}」を含む節・条文を持つ。",
    "HAS_ITEM":     "{src}において「{tgt}」という項目が規定されている。",
    "DESCRIBED_IN": "「{src}」に関する技術基準は「{tgt}」に記述されている。",
    "SUBJECT_TO":   "「{src}」は「{tgt}」の影響を受けるリスクがある。",
    "MITIGATES":    "「{src}」は「{tgt}」を軽減・防止する機能を持つ。",
    "REQUIRES":     "「{src}」の設計・管理には「{tgt}」の技術概念が必要とされる。",
    "DEFINED_IN":   "「{src}」という技術概念は「{tgt}」で定義・解説されている。",
    "USED_IN":      "「{src}」は「{tgt}」のプロセスで活用される。",
    "PRECEDES":     "「{src}」は「{tgt}」に先行して実施される工程である。",
    "AFFECTS":      "「{src}」というハザードは「{tgt}」に影響を与える。",
}

SYSTEM_PROMPT = """\
あなたは「河川砂防技術基準」を熟知した技術者教育の専門家です。
与えられた知識グラフのリレーション情報をもとに、現場エンジニアが実務で
重要と感じる質問と模範解答のペアを JSON 配列で{n}問生成してください。

制約:
1. 質問は「なぜ」「どのように」「いつ」「何を根拠に」など実務目線で具体的に
2. 回答は技術基準の趣旨に沿い 150〜350字程度（箇条書き可）
3. 異なる観点（定義・手順・比較・適用条件・ハザード対応など）で多様に出題
4. 推測や基準外の情報を含めない
5. 出力は JSON 配列のみ（前後の説明文は不要）:
   [{{"question": "...", "answer": "..."}}, ...]
"""

# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────
def load_nodes(csvs: list[str]) -> dict[str, dict]:
    """全ノードを id → {name, label, description} で返す。"""
    nodes: dict[str, dict] = {}
    for path in csvs:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            nid   = row.get("id:ID", "")
            name  = row.get("name", nid)
            label = row.get(":LABEL", "")
            desc  = row.get("description", "")
            nodes[nid] = {"name": name, "label": label, "description": desc}
    return nodes


def load_relations(path: str) -> list[tuple[str, str, str]]:
    rels = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        rels.append((row[":START_ID"], row[":END_ID"], row[":TYPE"]))
    return rels


def build_context(src: dict, tgt: dict, rel_type: str) -> str:
    tmpl = REL_TEMPLATE.get(rel_type, "「{src}」と「{tgt}」は {rel} の関係にある。")
    ctx  = tmpl.format(src=src["name"], tgt=tgt["name"], rel=rel_type)
    parts = [ctx]
    if src.get("description"):
        parts.append(f"※{src['name']}: {src['description']}")
    if tgt.get("description"):
        parts.append(f"※{tgt['name']}: {tgt['description']}")
    return "\n".join(parts)


def bigram_set(text: str) -> set:
    t = re.sub(r"\s+", "", text)
    return {t[i:i+2] for i in range(len(t) - 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = bigram_set(a), bigram_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_test_questions(path: str) -> list[str]:
    qs = json.load(open(path, encoding="utf-8"))
    return [q["question"] for q in qs]


def is_too_similar(question: str, test_qs: list[str], thresh: float) -> bool:
    return any(jaccard(question, tq) >= thresh for tq in test_qs)


# ──────────────────────────────────────────────
# Ollama 呼び出し
# GPT-OSS Swallow は /api/generate + raw=True + チャンネルトークン形式が必要
# ──────────────────────────────────────────────
def call_ollama(context: str, model: str, n: int) -> list[dict]:
    system = SYSTEM_PROMPT.format(n=n)
    user   = f"【リレーション情報】\n{context}"

    # GPT-OSS Swallow のチャンネルトークン形式
    prompt = (
        f"<|start|>system<|message|>{system}<|end|>\n"
        f"<|start|>user<|message|>{user}<|end|>\n"
        "<|start|>assistant<|channel|>final<|message|>"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "temperature":    0.7,
            "repeat_penalty": 1.2,
            "num_ctx":        4096,
            "num_predict":    1024,
            "stop":           ["<|end|>", "<|start|>"],
        },
    }
    for attempt in range(RETRY + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json().get("response", "").strip()

            # JSON 配列を抽出（```json ... ``` ブロックも対応）
            m = re.search(r"\[.*?\]", content, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 配列なし: {content[:120]}")
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [WARN] JSON パースエラー (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"    [ERROR] Ollama エラー (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []


# ──────────────────────────────────────────────
# Alpaca 形式変換
# ──────────────────────────────────────────────
def to_alpaca(qa: dict, src_name: str, tgt_name: str, rel_type: str) -> dict:
    return {
        "instruction": qa.get("question", "").strip(),
        "input": "",
        "output": qa.get("answer", "").strip(),
        "metadata": {
            "source": "graph_relation",
            "rel_type": rel_type,
            "src": src_name,
            "tgt": tgt_name,
        },
    }


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--n_per_rel",   type=int, default=N_PER_REL)
    parser.add_argument("--sim_thresh",  type=float, default=SIMILARITY_THRESH)
    parser.add_argument("--delay",       type=float, default=INTER_DELAY)
    parser.add_argument("--dry_run",     action="store_true", help="文脈確認のみ、LLM 呼び出しなし")
    parser.add_argument("--out",         default=str(OUT_DIR / "train_graph_rels.jsonl"))
    args = parser.parse_args()

    print("=== LoRA 学習用 QA 生成 (グラフリレーション駆動) ===")
    print(f"  モデル    : {args.model}")
    print(f"  リレーション: {RELATION_CSV}")
    print(f"  1件あたり : {args.n_per_rel}問")
    print(f"  出力      : {args.out}")

    nodes    = load_nodes(NODE_CSVS)
    rels     = load_relations(RELATION_CSV)
    test_qs  = load_test_questions(TEST_QUESTIONS)

    print(f"  ノード数  : {len(nodes)}")
    print(f"  リレーション数: {len(rels)}")
    print(f"  テスト問題数  : {len(test_qs)} (独立性フィルタ用)")
    print(f"  目標生成数    : {len(rels)} × {args.n_per_rel} = {len(rels)*args.n_per_rel}問")

    if args.dry_run:
        print("\n--- DRY RUN: 最初の5リレーションの文脈 ---")
        for sid, eid, rtype in rels[:5]:
            src = nodes.get(sid, {"name": sid, "label": "", "description": ""})
            tgt = nodes.get(eid, {"name": eid, "label": "", "description": ""})
            print(f"\n[{rtype}] {src['name']} → {tgt['name']}")
            print(build_context(src, tgt, rtype))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    dataset: list[dict] = []
    skipped_sim  = 0
    skipped_empty = 0

    for i, (sid, eid, rtype) in enumerate(rels, 1):
        src = nodes.get(sid, {"name": sid, "label": "", "description": ""})
        tgt = nodes.get(eid, {"name": eid, "label": "", "description": ""})
        context = build_context(src, tgt, rtype)

        print(f"\n[{i:03d}/{len(rels)}] {rtype}: {src['name']} → {tgt['name']}")

        qa_list = call_ollama(context, args.model, args.n_per_rel)

        accepted = 0
        for qa in qa_list:
            q = qa.get("question", "").strip()
            a = qa.get("answer",   "").strip()
            if not q or not a:
                skipped_empty += 1
                continue
            if is_too_similar(q, test_qs, args.sim_thresh):
                print(f"    [SKIP] 類似度超過: {q[:40]}...")
                skipped_sim += 1
                continue
            dataset.append(to_alpaca(qa, src["name"], tgt["name"], rtype))
            accepted += 1

        print(f"    採用 {accepted}/{len(qa_list)} 問（累計 {len(dataset)} 問）")
        time.sleep(args.delay)

    # ──────────────────────────────────────────
    # 保存
    # ──────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== 完了 ===")
    print(f"  採用   : {len(dataset)} 問")
    print(f"  除外   : 類似度超過 {skipped_sim}問 / 空 {skipped_empty}問")
    print(f"  出力   : {out_path}")

    # 簡易サマリー（リレーション種別ごと）
    from collections import Counter
    cnt = Counter(r["metadata"]["rel_type"] for r in dataset)
    print("\n  種別内訳:")
    for rtype, n in cnt.most_common():
        print(f"    {rtype:<20}: {n:>3}問")


if __name__ == "__main__":
    main()
