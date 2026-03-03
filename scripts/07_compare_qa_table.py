"""
scripts/07_compare_qa_table.py
A / B / C 代表10問 Q&A 定性比較表を Markdown 形式で出力する

出力フォーマット（問ごとに 4行2列テーブル）:
| 列1              | 列2                        |
|------------------|----------------------------|
| Question         | Question Prompt            |
| Plain LLM        | Response A (Swallow-20B)   |
| QLoRA Fine-Tune  | Response B (Swallow-8B FT) |
| GraphRAG         | Response C (Swallow-20B)   |
"""
import json
import re

# ── 選定問題 ID（10問） ───────────────────────────────────────────────────
SELECTED_IDS = [5, 14, 24, 26, 37, 52, 69, 82, 91, 95]

# 選定理由メモ（コメント用）
ID_NOTES = {
    5:  "B-A= +3 | A=0,B=3,C=3 | 堤防長寿命化（維持管理_河川）",
    14: "全員=3  | A=3,B=3,C=3 | サイクル型維持管理体系（維持管理_河川）",
    24: "B-A= +3 | A=0,B=3,C=3 | ダム長寿命化計画（維持管理_ダム）",
    26: "A>B,C   | A=3,B=2,C=2 | 堆砂率の計算方法（維持管理_ダム）",
    37: "B=3,C=0 | A=1,B=3,C=0 | 土石流後の砂防堰堤臨時点検（維持管理_砂防）",
    52: "A>B>C   | A=3,B=2,C=1 | 流域平均雨量算定法（調査）",
    69: "B-A= +2 | A=1,B=3,C=3 | 砂防堰堤の安定計算（設計）",
    82: "B-A= +3 | A=0,B=3,C=3 | コンクリートvsフィルダム安定計算比較（比較・横断）",
    91: "B-A= +2 | A=1,B=3,C=3 | 計画規模洪水に対する堤防安全確保（ハザード）",
    95: "A>B>C   | A=3,B=2,C=1 | 地すべり活動度評価と対策優先度（ハザード）",
}

# ── データ読み込み ────────────────────────────────────────────────────────
with open("data/eval/results/results_20260301_210818.jsonl", "r", encoding="utf-8") as f:
    recs_ac = {json.loads(l)["id"]: json.loads(l) for l in f}

with open("data/eval/results/results_b_20260302_214650.jsonl", "r", encoding="utf-8") as f:
    recs_b = {json.loads(l)["id"]: json.loads(l) for l in f}

# ── 出力ヘルパー ─────────────────────────────────────────────────────────
def to_cell(text: str) -> str:
    """回答テキストを Markdown テーブルセル用に変換する。
    改行 → <br>、パイプ文字 → &#124; でエスケープ。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text.replace("|", "&#124;")
    text = text.replace("\n", "<br>")
    return text

# ── Markdown 生成 ─────────────────────────────────────────────────────────
lines = []
lines.append("# A / B / C 代表 10問 Q&A 定性比較表\n")
lines.append("> | Case | 構成 | モデル |")
lines.append("> |---|---|---|")
lines.append("> | **Plain LLM** (A) | Raw LLM, no RAG | GPT-OSS Swallow-20B-RL (Q4_K_M) |")
lines.append("> | **QLoRA Fine-Tune** (B) | LoRA FT only | Swallow-8B-Instruct, n=715, Q4_K_M |")
lines.append("> | **GraphRAG** (C) | GraphRAG + Raw LLM | GPT-OSS Swallow-20B-RL (Q4_K_M) |")
lines.append(">")
lines.append("> **Judge**: Qwen2.5-14B / Scoring rubric: 0–3  ")
lines.append("> **Full answer texts**: see code blocks at bottom of each table.  ")
lines.append("")

for qid in SELECTED_IDS:
    rec_ac = recs_ac.get(qid)
    rec_b  = recs_b.get(qid)
    if not rec_ac or not rec_b:
        print(f"WARNING: Q{qid} のデータが見つかりません")
        continue

    q      = rec_ac["question"]
    cat    = rec_ac["category"]
    subcat = rec_ac.get("subcategory", "")
    note   = ID_NOTES.get(qid, "")

    ca = rec_ac["case_a"]
    cb = rec_b["case_b"]
    cc = rec_ac["case_c"]

    sa = ca["judge"]["score"]
    sb = cb["judge"]["score"]
    sc = cc["judge"]["score"]

    ra = ca["judge"]["reason"]
    rb = cb["judge"]["reason"]
    rc = cc["judge"]["reason"]

    ans_a = to_cell(ca["answer"])
    ans_b = to_cell(cb["answer"])
    ans_c = to_cell(cc["answer"])

    # ── セクションヘッダ ─────────────────────────────────────────────────
    lines.append(f"---\n")
    lines.append(f"### Q{qid}. {q}")
    lines.append(f"")
    lines.append(f"Category: **{cat}** / {subcat} &nbsp;｜&nbsp; {note}")
    lines.append(f"")

    # ── 4行 × 2列テーブル ────────────────────────────────────────────────
    lines.append(f"| | Response |")
    lines.append(f"|---|---|")
    lines.append(f"| **Question** | {q} |")
    lines.append(f"| **Plain LLM** (A)<br>*Swallow-20B*<br>Score **{sa}**/3 &nbsp; {ca['elapsed_s']:.1f}s | {ans_a} |")
    lines.append(f"| **QLoRA Fine-Tune** (B)<br>*Swallow-8B FT*<br>Score **{sb}**/3 &nbsp; {cb['elapsed_s']:.1f}s | {ans_b} |")
    lines.append(f"| **GraphRAG** (C)<br>*Swallow-20B*<br>Score **{sc}**/3 &nbsp; {cc['elapsed_s']:.1f}s | {ans_c} |")
    lines.append(f"")

    # ── Judge 評価 ────────────────────────────────────────────────────────
    lines.append(f"> **Judge A**: {ra}  ")
    lines.append(f"> **Judge B**: {rb}  ")
    lines.append(f"> **Judge C**: {rc}  ")
    lines.append(f"")

# サマリテーブル
lines.append("---\n")
lines.append("## 選定 10問 スコアサマリ\n")
lines.append("| Q# | カテゴリ | サブカテゴリ | Plain LLM (A) | QLoRA FT (B) | GraphRAG (C) | B−A | パターン |")
lines.append("|---|---|---|:---:|:---:|:---:|:---:|---|")
for qid in SELECTED_IDS:
    rec_ac = recs_ac.get(qid)
    rec_b  = recs_b.get(qid)
    if not rec_ac or not rec_b:
        continue
    sa = rec_ac["case_a"]["judge"]["score"]
    sb = rec_b["case_b"]["judge"]["score"]
    sc = rec_ac["case_c"]["judge"]["score"]
    diff = sb - sa
    note = ID_NOTES.get(qid, "")
    pattern = note.split("|")[0].strip() if "|" in note else ""
    lines.append(f"| Q{qid} | {rec_ac['category']} | {rec_ac.get('subcategory','')} | {sa} | {sb} | {sc} | {diff:+d} | {pattern} |")

# 出力
out_path = "docs/qa_comparison_10q.md"
import os
os.makedirs("docs", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ 出力完了: {out_path} ({len(lines)} 行)")
print(f"\n=== スコアサマリ ===")
print(f"{'Q':4s} | {'Cat':14s} | A | B | C | B-A")
print("-" * 50)
for qid in SELECTED_IDS:
    rec_ac = recs_ac.get(qid)
    rec_b  = recs_b.get(qid)
    if not rec_ac or not rec_b:
        continue
    sa = rec_ac["case_a"]["judge"]["score"]
    sb = rec_b["case_b"]["judge"]["score"]
    sc = rec_ac["case_c"]["judge"]["score"]
    diff = sb - sa
    print(f"Q{qid:3d} | {rec_ac['category']:14s} | {sa} | {sb} | {sc} | {diff:+d}")
