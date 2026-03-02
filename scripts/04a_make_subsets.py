"""
04a_make_subsets.py
────────────────────────────────────────────────────────────
data/lora/train_graph_rels.jsonl (715問) から
rel_type 層別サンプリングで 4段階のサブセットを生成する。

出力:
  data/lora/subset_100.jsonl
  data/lora/subset_250.jsonl
  data/lora/subset_500.jsonl
  data/lora/subset_715.jsonl  ← 全量コピー

Usage:
    python scripts/04a_make_subsets.py
    python scripts/04a_make_subsets.py --seed 99
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
SRC_PATH  = Path("data/lora/train_graph_rels.jsonl")
OUT_DIR   = Path("data/lora")
SUBSETS   = [100, 250, 500, 715]
SEED      = 42


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  保存: {path}  ({len(records)} 問)")


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """
    rel_type 層別サンプリング。
    各クラスから n × (クラス割合) 問を抽出し、合計を n に合わせる。
    """
    if n >= len(records):
        return list(records)

    # クラス別に分類
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        rel = r.get("metadata", {}).get("rel_type", "UNKNOWN")
        buckets[rel].append(r)

    rng = random.Random(seed)
    total = len(records)
    sampled: list[dict] = []

    # 各クラスの割り当て数を計算（比例配分 + 端数調整）
    alloc: dict[str, int] = {}
    for rel, items in buckets.items():
        alloc[rel] = max(1, math.floor(n * len(items) / total))

    # 合計が n になるよう端数を大クラスに追加
    diff = n - sum(alloc.values())
    if diff > 0:
        sorted_rels = sorted(buckets.keys(), key=lambda r: -len(buckets[r]))
        for rel in sorted_rels:
            if diff == 0:
                break
            alloc[rel] += 1
            diff -= 1

    # 各クラスからサンプリング
    for rel, k in alloc.items():
        items = buckets[rel]
        sampled.extend(rng.sample(items, min(k, len(items))))

    # シャッフル
    rng.shuffle(sampled)
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  default=str(SRC_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[ERROR] 元データが見つかりません: {src}")
        print("  先に scripts/03b_generate_lora_qa_graph.py を実行してください。")
        return

    records = load_jsonl(src)
    print(f"=== サブセット生成 (seed={args.seed}) ===")
    print(f"  元データ: {len(records)} 問  ({src})")

    # rel_type 分布を表示
    from collections import Counter
    cnt = Counter(r.get("metadata", {}).get("rel_type", "?") for r in records)
    print("\n  rel_type 分布:")
    for rel, n in cnt.most_common():
        print(f"    {rel:<20}: {n:>3}問  ({n/len(records)*100:.1f}%)")

    print()
    for size in SUBSETS:
        subset = stratified_sample(records, size, seed=args.seed)
        out_path = OUT_DIR / f"subset_{size}.jsonl"
        save_jsonl(subset, out_path)

        # 確認: rel_type 分布
        sub_cnt = Counter(r.get("metadata", {}).get("rel_type", "?") for r in subset)
        print(f"    rel_type 内訳: " + ", ".join(f"{r}:{n}" for r, n in sub_cnt.most_common(4)))

    print("\n=== 完了 ===")
    print("  生成ファイル:")
    for size in SUBSETS:
        p = OUT_DIR / f"subset_{size}.jsonl"
        print(f"    {p}")


if __name__ == "__main__":
    main()
