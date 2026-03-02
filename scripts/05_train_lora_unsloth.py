"""
05_train_lora_unsloth.py
────────────────────────────────────────────────────────────
GPT-OSS-Swallow-8B-Instruct を unsloth QLoRA でファインチューニング。
4段階のサブセット（100 / 250 / 500 / 715問）で安定収束を確認する。

前提:
  pip install "unsloth[cocu-ampere-cu124]" trl transformers datasets accelerate

Usage:
    # 単一サブセットで学習
    python scripts/05_train_lora_unsloth.py --subset 100

    # 全 4 段階を順番に実行
    python scripts/05_train_lora_unsloth.py --subset all

    # 学習済みアダプタを GGUF にエクスポート
    python scripts/05_train_lora_unsloth.py --subset 715 --export_gguf

出力:
    models/lora/swallow8b_n{N}/          ← LoRA アダプタ
    models/gguf/swallow8b_lora_n{N}/     ← GGUF (--export_gguf 時)
    data/lora/train_loss_{N}.json        ← 学習ロス履歴
"""

import argparse
import json
import time
from pathlib import Path

# ── 必要なライブラリ確認 ──
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
except ImportError as e:
    print(f"[ERROR] 必要なライブラリが不足しています: {e}")
    print("  pip install \"unsloth[cu124-ampere]\" trl transformers datasets accelerate")
    raise SystemExit(1)

# ──────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────
BASE_MODEL      = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
MAX_SEQ_LEN     = 2048
LOAD_4BIT       = True

LORA_R          = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.0
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_EPOCHS        = 3
PER_DEVICE_BATCH    = 2
GRAD_ACCUMULATION   = 4          # 実効バッチサイズ = 2 × 4 = 8
LEARNING_RATE       = 2e-4
LR_SCHEDULER        = "cosine"
WARMUP_RATIO        = 0.05
WEIGHT_DECAY        = 0.01

SUBSETS_DIR  = Path("data/lora")
MODELS_DIR   = Path("models/lora")
GGUF_DIR     = Path("models/gguf")
LOSS_DIR     = Path("data/lora")

SYSTEM_PROMPT = (
    "あなたは河川砂防技術基準（調査・計画・設計・維持管理）を熟知した"
    "専門家です。正確で実務的な回答をしてください。"
)

# ──────────────────────────────────────────────
# Llama-3-Swallow-8B-Instruct プロンプトフォーマット
# Meta Llama-3 Instruct 形式
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
)


def format_record(record: dict) -> dict:
    """Alpaca 形式レコード → Swallow-8B-Instruct 学習テキストに変換。"""
    instruction = record.get("instruction", "").strip()
    context     = record.get("input", "").strip()
    output      = record.get("output", "").strip()

    # コンテキストがある場合は instruction に結合
    if context:
        user_part = f"{instruction}\n\n【参考情報】\n{context}"
    else:
        user_part = instruction

    text = PROMPT_TEMPLATE.format(
        system      = SYSTEM_PROMPT,
        instruction = user_part,
        output      = output,
    )
    return {"text": text}


# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────
def load_subset(size: int) -> Dataset:
    path = SUBSETS_DIR / f"subset_{size}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。\n"
            "  先に scripts/04a_make_subsets.py を実行してください。"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    formatted = [format_record(r) for r in records]
    dataset   = Dataset.from_list(formatted)
    print(f"  データ読み込み完了: {len(dataset)} 件  ({path})")
    return dataset


# ──────────────────────────────────────────────
# LoRA 学習
# ──────────────────────────────────────────────
def train_one(size: int, export_gguf: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  LoRA 学習開始: subset={size}問  base={BASE_MODEL}")
    print(f"{'='*60}")

    output_dir  = MODELS_DIR / f"swallow8b_n{size}"
    loss_path   = LOSS_DIR   / f"train_loss_{size}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── モデル & トークナイザー読み込み ──
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = MAX_SEQ_LEN,
        dtype           = None,       # 自動判定 (bfloat16 推奨)
        load_in_4bit    = LOAD_4BIT,
    )

    # ── LoRA アダプタ付与 ──
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_R,
        target_modules      = TARGET_MODULES,
        lora_alpha          = LORA_ALPHA,
        lora_dropout        = LORA_DROPOUT,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",  # VRAM 節約
        random_state        = 42,
        use_rslora          = False,
        loftq_config        = None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  学習可能パラメータ: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── データセット ──
    dataset = load_subset(size)

    # ── SFTTrainer 設定 ──
    sft_config = SFTConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = TRAIN_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        warmup_ratio                = WARMUP_RATIO,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = max(1, len(dataset) // (PER_DEVICE_BATCH * GRAD_ACCUMULATION * 5)),
        save_strategy               = "epoch",
        save_total_limit            = 1,
        seed                        = 42,
        report_to                   = "none",
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LEN,
        packing                     = True,   # 短いサンプルを連結して GPU 効率化
    )

    # ── ロスコールバック ──
    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def __init__(self):
            self.history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.history.append({
                    "step":  state.global_step,
                    "epoch": round(state.epoch, 3),
                    "loss":  round(logs["loss"], 4),
                })

    loss_logger = LossLogger()

    trainer = SFTTrainer(
        model       = model,
        tokenizer   = tokenizer,
        train_dataset = dataset,
        args        = sft_config,
        callbacks   = [loss_logger],
    )

    # ── 学習実行 ──
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n  学習完了: {elapsed/60:.1f} 分")

    # ── アダプタ保存 ──
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  アダプタ保存: {output_dir}")

    # ── ロス履歴保存 ──
    loss_data = {
        "subset":   size,
        "model":    BASE_MODEL,
        "epochs":   TRAIN_EPOCHS,
        "elapsed_min": round(elapsed / 60, 1),
        "history":  loss_logger.history,
        "final_loss": loss_logger.history[-1]["loss"] if loss_logger.history else None,
    }
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(loss_data, f, ensure_ascii=False, indent=2)
    print(f"  ロス履歴保存: {loss_path}")
    if loss_logger.history:
        print(f"  最終ロス: {loss_logger.history[-1]['loss']:.4f} (step {loss_logger.history[-1]['step']})")

    # ── GGUF エクスポート（オプション） ──
    if export_gguf:
        gguf_out = GGUF_DIR / f"swallow8b_lora_n{size}"
        gguf_out.mkdir(parents=True, exist_ok=True)
        print(f"\n  GGUF エクスポート中 → {gguf_out}")
        # unsloth は save_pretrained_gguf で直接 GGUF 出力可能
        model.save_pretrained_gguf(
            str(gguf_out),
            tokenizer,
            quantization_method = "q4_k_m",  # Ollama 標準量子化
        )
        # Ollama 用 Modelfile 生成
        modelfile_path = gguf_out / "Modelfile"
        gguf_files = list(gguf_out.glob("*.gguf"))
        gguf_name  = gguf_files[0].name if gguf_files else "model.gguf"
        modelfile_path.write_text(
            f'FROM ./{gguf_name}\n'
            f'SYSTEM "{SYSTEM_PROMPT}"\n'
            f'PARAMETER temperature 0.3\n'
            f'PARAMETER repeat_penalty 1.1\n',
            encoding="utf-8",
        )
        print(f"  Modelfile 生成: {modelfile_path}")
        print(f"\n  Ollama へのロード:")
        print(f"    cd {gguf_out}")
        print(f"    ollama create swallow8b-lora-n{size} -f Modelfile")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    global BASE_MODEL  # 関数先頭で宣言

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["100", "250", "500", "715", "all"],
        default="all",
        help="学習するサブセットサイズ (all で 4 段階を順番に実行)",
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="学習後に GGUF (Q4_K_M) へエクスポートする",
    )
    parser.add_argument(
        "--base_model",
        default=BASE_MODEL,
        help=f"ベースモデル HF ID (デフォルト: {BASE_MODEL})",
    )
    args = parser.parse_args()

    # ベースモデル上書き
    BASE_MODEL = args.base_model

    if args.subset == "all":
        sizes = [100, 250, 500, 715]
    else:
        sizes = [int(args.subset)]

    print("=== Swallow-8B-Instruct QLoRA 学習 ===")
    print(f"  ベースモデル : {args.base_model}")
    print(f"  サブセット   : {sizes}")
    print(f"  GGUF 出力    : {args.export_gguf}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, epoch={TRAIN_EPOCHS}")
    print(f"  バッチ={PER_DEVICE_BATCH} × accum={GRAD_ACCUMULATION} = {PER_DEVICE_BATCH*GRAD_ACCUMULATION} (実効)")

    for size in sizes:
        train_one(size, export_gguf=args.export_gguf)

    print("\n=== 全学習完了 ===")
    print("  収束比較:")
    for size in sizes:
        loss_path = LOSS_DIR / f"train_loss_{size}.json"
        if loss_path.exists():
            data = json.load(open(loss_path, encoding="utf-8"))
            fl   = data.get("final_loss", "N/A")
            et   = data.get("elapsed_min", "?")
            print(f"    subset={size:>3}問: 最終ロス={fl}  学習時間={et}分")


if __name__ == "__main__":
    main()
