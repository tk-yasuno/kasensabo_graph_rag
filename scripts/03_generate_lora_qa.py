"""
03_generate_lora_qa.py
────────────────────────────────────────────────────────────
河川砂防技術基準 Markdown → LoRA 学習用 QA ペア自動生成

出力: JSONL（instruction / input / output 形式）
  → GPT-OSS 8B + QLoRA / Axolotl / LLaMA-Factory で使用可

Usage:
    # 1ファイル指定
    python scripts/03_generate_lora_qa.py \
        --input data/kasen-dam-sabo_Train_set/05_training_ijikanri_kasen_2025.md \
        --output data/lora/train_ijikanri_kasen.jsonl \
        --target 200

    # 全ファイル一括（デフォルト実行）
    python scripts/03_generate_lora_qa.py

環境変数:
    OPENAI_API_KEY
"""

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

client = OpenAI()

# ──────────────────────────────────────────────
# 質問タイプのテンプレート（プロンプト多様化用）
# ──────────────────────────────────────────────
QUESTION_TYPES = [
    "定義・目的を説明する質問",
    "手順・方法を説明する質問",
    "施設・概念を比較する質問",
    "適用条件・例外を問う質問",
    "技術基準の適用区分（必須/標準/推奨）に関する質問",
    "実務者が現場で直面する判断を問う質問",
    "ハザード（洪水・土石流・地すべり）と施設の関係を問う質問",
    "維持管理計画・点検手順を問う質問",
]

SYSTEM_PROMPT = """\
あなたは「河川砂防技術基準」を熟知した技術教育専門家です。
与えられたテキストを読み、現場の河川・砂防エンジニアが実務で遭遇しそうな
質問と模範解答のペアを JSON 配列で生成してください。

制約:
1. 質問は具体的かつ実務レベル（「なぜ」「どのように」「いつ」「何を」が明確）
2. 回答は技術基準の趣旨に沿い、必須/標準/推奨の区別を意識する
3. 回答は 150〜400 字程度（箇条書き可）
4. テキストに明記されていない推測を回答に含めない
5. 出力は JSON 配列のみ（説明文不要）:
   [{"question": "...", "answer": "..."}, ...]
"""

# ──────────────────────────────────────────────
# チャンク化
# ──────────────────────────────────────────────
def chunk_by_heading(text: str, max_len: int = 1500) -> list[dict]:
    chunks: list[dict] = []
    lines = text.split("\n")
    current: list[str] = []
    header = "本文"

    for line in lines:
        if re.match(r"^#{1,4}\s", line):
            if current:
                chunks.append({"header": header, "text": "\n".join(current).strip()})
                current = []
            header = line.lstrip("#").strip()
        current.append(line)
        if sum(len(l) for l in current) >= max_len:
            chunks.append({"header": header, "text": "\n".join(current).strip()})
            current = []

    if current:
        chunks.append({"header": header, "text": "\n".join(current).strip()})

    return [c for c in chunks if len(c["text"]) > 100]


# ──────────────────────────────────────────────
# QA 生成（chunk 単位）
# ──────────────────────────────────────────────
def generate_qa(
    chunk: dict,
    model: str = "gpt-4o-mini",
    n_per_chunk: int = 5,
    retry: int = 3,
) -> list[dict]:
    q_types = ", ".join(QUESTION_TYPES[:n_per_chunk])
    user_msg = (
        f"[見出し: {chunk['header']}]\n\n"
        f"以下の質問タイプを参考に {n_per_chunk} 問を作成してください。\n"
        f"質問タイプ例: {q_types}\n\n"
        f"【テキスト】\n{chunk['text']}"
    )

    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            # 配列または {"qa": [...]} 形式に対応
            if isinstance(data, list):
                return data
            for key in ("qa", "questions", "qa_pairs", "items"):
                if key in data:
                    return data[key]
            return list(data.values())[0] if data else []
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON パースエラー (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] API エラー (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []


# ──────────────────────────────────────────────
# JSONL 書き出し
# ──────────────────────────────────────────────
def to_instruction_format(qa: dict, source_file: str, header: str) -> dict:
    """QA ペアを Alpaca instruction 形式に変換。"""
    return {
        "instruction": qa.get("question", "").strip(),
        "input": "",
        "output": qa.get("answer", "").strip(),
        "metadata": {
            "source": source_file,
            "section": header,
        },
    }


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
def build_dataset(
    input_path: str,
    output_path: str,
    target_size: int = 200,
    model: str = "gpt-4o-mini",
    n_per_chunk: int = 5,
    chunk_size: int = 1500,
    delay: float = 0.8,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"\n=== QA 生成: {input_path.name} → {output_path.name} ===")
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_by_heading(text, max_len=chunk_size)
    print(f"  チャンク数: {len(chunks)}  目標サンプル数: {target_size}")

    dataset: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        if len(dataset) >= target_size:
            break
        print(f"  [{i}/{len(chunks)}] {chunk['header'][:50]!r}")
        qa_list = generate_qa(chunk, model=model, n_per_chunk=n_per_chunk)
        for qa in qa_list:
            if not qa.get("question") or not qa.get("answer"):
                continue
            dataset.append(
                to_instruction_format(qa, input_path.name, chunk["header"])
            )
            if len(dataset) >= target_size:
                break
        time.sleep(delay)

    # 空フィルタ
    dataset = [d for d in dataset if d["instruction"] and d["output"]]
    write_jsonl(dataset, output_path)
    print(f"=== 完了: {len(dataset)} サンプル → {output_path} ===\n")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="LoRA 学習用 QA ペア生成")
    parser.add_argument("--input",  help="入力 Markdown ファイルパス")
    parser.add_argument("--output", help="出力 JSONL ファイルパス")
    parser.add_argument("--target", type=int, default=200, help="目標サンプル数")
    parser.add_argument("--model",  default="gpt-4o-mini", help="OpenAI モデル")
    parser.add_argument("--n_per_chunk", type=int, default=5, help="チャンクあたり QA 数")
    parser.add_argument("--chunk_size",  type=int, default=1500, help="チャンク最大文字数")
    parser.add_argument("--delay", type=float, default=0.8, help="API 間隔(秒)")
    args = parser.parse_args()

    if args.input and args.output:
        build_dataset(
            input_path=args.input,
            output_path=args.output,
            target_size=args.target,
            model=args.model,
            n_per_chunk=args.n_per_chunk,
            chunk_size=args.chunk_size,
            delay=args.delay,
        )
    else:
        # デフォルト: 全ファイル処理（ファイルごとに出力）
        train_dir = Path("data/kasen-dam-sabo_Train_set")
        out_dir   = Path("data/lora")

        # ファイルごとの生成目標数
        targets = {
            "00_training_overview_2025.md":          30,
            "01_training_chousa_2025.md":            50,
            "02_training_keikaku_kihon_2025.md":     50,
            "03_training_keikaku_shisetsu_2025.md":  50,
            "04_training_sekkei_2025.md":            80,
            "05_training_ijikanri_kasen_2025.md":   100,
            "06_training_ijikanri_dam_2025.md":      80,
            "07_training_ijikanri_sabo_2025.md":     60,
        }

        all_records: list[dict] = []
        for fname, n in targets.items():
            md_path = train_dir / fname
            if not md_path.exists():
                print(f"  [SKIP] {fname} が見つかりません")
                continue
            stem = md_path.stem
            out_path = out_dir / f"qa_{stem}.jsonl"
            build_dataset(
                input_path=str(md_path),
                output_path=str(out_path),
                target_size=n,
            )
            # 個別ファイルも読み込んで全体マージ用に蓄積
            with open(out_path, encoding="utf-8") as f:
                all_records.extend([json.loads(l) for l in f if l.strip()])

        # 全データをシャッフルして train / val に分割
        import random
        random.seed(42)
        random.shuffle(all_records)
        split = int(len(all_records) * 0.9)
        write_jsonl(all_records[:split],  out_dir / "train_all.jsonl")
        write_jsonl(all_records[split:],  out_dir / "val_all.jsonl")
        print(f"\n=== 全体マージ完了: train={split} / val={len(all_records)-split} ===")


if __name__ == "__main__":
    import sys
    main()
