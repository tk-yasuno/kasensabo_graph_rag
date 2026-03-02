"""
A/B/C アプローチ比較 — 3図を生成して docs/figures/ に保存する。

生成図:
  fig1_quadrant.png   : 推論速度 vs 回答精度（散布図＋象限）
  fig2_scores.png     : Judge スコア棒グラフ（全体 + スコア分布）
  fig3_evolution.png  : アプローチ進化タイムライン（横方向）
"""

import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic",
                                       "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = pathlib.Path(__file__).parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── データ ──────────────────────────────────────────────────────────────
CASES = ["Case A\n20B Plain", "Case B\n8B LoRA FT", "Case C\n20B GraphRAG"]
COLORS = ["#5b9bd5", "#ed7d31", "#70ad47"]   # blue / orange / green

LATENCY  = [42.2, 14.2, 31.1]   # seconds
SCORES   = [2.29, 2.92, 2.62]   # Judge avg /3

DIST = {          # score distribution (%)
    "Case A": [3, 25, 12, 60],    # 0pt, 1pt, 2pt, 3pt
    "Case B": [0,  0,  8, 92],
    "Case C": [2, 11, 10, 77],
}

# ── 図1: クアドラント（散布図） ─────────────────────────────────────────
def plot_quadrant():
    lat_max, lat_min = max(LATENCY), min(LATENCY)
    speed_norm = [(lat_max - l) / (lat_max - lat_min) for l in LATENCY]
    acc_norm   = [(s - 2.0) / 1.0 for s in SCORES]   # 2.0〜3.0 → 0〜1

    fig, ax = plt.subplots(figsize=(6, 6))

    # 象限の背景色
    ax.axhspan(0.5, 1.05, xmin=0.5, xmax=1.05, color="#d9f0cb", alpha=0.4, zorder=0)
    ax.axhspan(-0.05, 0.5, xmin=-0.05, xmax=0.5, color="#fce4d6", alpha=0.3, zorder=0)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axvline(0.5, color="gray", lw=0.8, ls="--")

    # データ点
    for i, (sx, sy, label, color) in enumerate(
            zip(speed_norm, acc_norm, CASES, COLORS)):
        ax.scatter(sx, sy, s=300, color=color, zorder=5, edgecolors="white", lw=1.5)
        offset = [(0.04, 0.04), (-0.18, 0.04), (0.04, -0.08)][i]
        ax.annotate(
            label, (sx, sy),
            xytext=(sx + offset[0], sy + offset[1]),
            fontsize=10, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85)
        )

    # 象限ラベル
    ax.text(0.75, 0.95, "理想ゾーン", ha="center", va="top",
            fontsize=9, color="#2e7d32", style="italic")
    ax.text(0.25, 0.95, "高精度\nだが低速", ha="center", va="top",
            fontsize=9, color="#555", style="italic")
    ax.text(0.75, 0.05, "高速\nだが低精度", ha="center", va="bottom",
            fontsize=9, color="#555", style="italic")
    ax.text(0.25, 0.05, "要改善", ha="center", va="bottom",
            fontsize=9, color="#c62828", style="italic")

    # 軸ラベル
    ax.set_xlabel("推論速度（遅い ← → 速い）", fontsize=11)
    ax.set_ylabel("回答精度   Judge avg /3", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["遅い\n(42s)", "中間\n(28s)", "速い\n(14s)"], fontsize=9)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["2.00", "2.50", "3.00"], fontsize=9)
    ax.set_title("① アプローチ比較：推論速度 vs 回答精度", fontsize=12,
                 fontweight="bold", pad=12)

    fig.tight_layout()
    out = OUT_DIR / "fig1_quadrant.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── 図2: Judge スコア（全体棒 + 分布積み上げ） ──────────────────────────
def plot_scores():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [1, 1.6]})

    # 左: 平均スコア棒グラフ
    short = ["Case A\n20B Plain", "Case B\n8B LoRA FT", "Case C\n20B GraphRAG"]
    bars = ax1.bar(short, SCORES, color=COLORS, edgecolor="white",
                   linewidth=1.5, width=0.5, zorder=3)
    ax1.set_ylim(0, 3.5)
    ax1.set_ylabel("Judge 平均スコア  (/3)", fontsize=11)
    ax1.set_title("平均スコア", fontsize=11, pad=8)
    ax1.axhline(3.0, color="gray", lw=0.8, ls="--")
    ax1.tick_params(labelsize=9)
    ax1.set_yticks([0, 1, 2, 2.29, 2.62, 2.92, 3])
    ax1.set_yticklabels(["0", "1", "2", "2.29", "2.62", "2.92", "3.0"],
                        fontsize=8)
    ax1.grid(axis="y", zorder=0, alpha=0.4)
    for bar, score in zip(bars, SCORES):
        ax1.text(bar.get_x() + bar.get_width() / 2, score + 0.05,
                 f"{score:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    # 差分アノテーション
    ax1.annotate("", xy=(1, SCORES[1]), xytext=(0, SCORES[0]),
                 arrowprops=dict(arrowstyle="->, head_width=0.2",
                                 color="#c62828", lw=1.5))
    ax1.text(0.5, (SCORES[0] + SCORES[1]) / 2 + 0.06, "+0.63",
             ha="center", color="#c62828", fontsize=9, fontweight="bold")

    # 右: スコア分布（積み上げ棒）
    score_labels = ["0点", "1点", "2点", "3点"]
    dist_colors  = ["#c62828", "#f4a261", "#2196f3", "#4caf50"]
    cases_short  = ["Case A", "Case B", "Case C"]
    x = np.arange(len(cases_short))
    bottom = np.zeros(3)
    for j, (lbl, col) in enumerate(zip(score_labels, dist_colors)):
        vals = [DIST[k][j] for k in DIST]
        ax2.bar(x, vals, bottom=bottom, label=lbl, color=col,
                edgecolor="white", linewidth=0.8, zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax2.text(xi, b + v / 2, f"{v}%", ha="center", va="center",
                         fontsize=9, fontweight="bold", color="white")
        bottom += np.array(vals)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Case A\n20B Plain", "Case B\n8B LoRA FT",
                          "Case C\n20B GraphRAG"], fontsize=9)
    ax2.set_ylabel("問数 (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("スコア分布（0〜3点）", fontsize=11, pad=8)
    ax2.legend(loc="upper right", fontsize=9, ncol=2)
    ax2.grid(axis="y", zorder=0, alpha=0.4)
    ax2.tick_params(labelsize=9)

    fig.suptitle("② Judge スコア比較（LLM-as-Judge Qwen2.5:14B / 100問）",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig2_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── 図3: アプローチ進化タイムライン ────────────────────────────────────
def plot_evolution():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")

    # ボックスの中心 x 座標・半幅
    BOX_HALF = 0.22
    MARGIN   = 0.06   # ボックス枠からの余白
    centers  = [0.0, 1.1, 2.2]

    stages = [
        ("Case A\nベースライン",
         "GPT-OSS Swallow 20B\nLatency: 42.2 s",
         "Judge avg\n2.29 / 3", "+0.00", COLORS[0], centers[0]),
        ("Case C\n知識グラフ強化",
         "GPT-OSS Swallow 20B\n+ Neo4j GraphRAG\nLatency: 31.1 s",
         "Judge avg\n2.62 / 3", "+0.33", COLORS[2], centers[1]),
        ("Case B\nLoRA FT (Best)",
         "Swallow-8B QLoRA FT\nn=715 pair\nLatency: 14.2 s",
         "Judge avg\n2.92 / 3", "+0.63", COLORS[1], centers[2]),
    ]

    # ① ボックスを先に描画
    for label, detail, score, delta, color, xc in stages:
        box = mpatches.FancyBboxPatch(
            (xc - BOX_HALF, 0.0), BOX_HALF * 2, 1.0,
            boxstyle="round,pad=0.04", linewidth=2,
            edgecolor=color, facecolor=color + "20", zorder=2
        )
        ax.add_patch(box)

        ax.text(xc, 0.9, label, ha="center", va="top",
                fontsize=10, fontweight="bold", color=color, zorder=3)
        ax.text(xc, 0.6, detail, ha="center", va="top",
                fontsize=8, color="#333", linespacing=1.5, zorder=3)
        ax.text(xc, 0.16, score, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, zorder=3)

        badge_color = "#4caf50" if delta != "+0.00" else "#aaa"
        ax.text(xc + 0.15, 1.05, delta, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none"))

    # ② 矢印をボックス後に描画（ボックス端からボックス端へ、余白 MARGIN 分あけて）
    for i in range(len(centers) - 1):
        x_start = centers[i]     + BOX_HALF + MARGIN   # 左ボックス右端 + 余白
        x_end   = centers[i + 1] - BOX_HALF - MARGIN   # 右ボックス左端 - 余白
        ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5),
                    arrowprops=dict(arrowstyle="->, head_width=0.12",
                                   color="gray", lw=2),
                    zorder=5)

    ax.set_title("④ アプローチ進化と精度向上の軌跡", fontsize=12,
                 fontweight="bold", pad=12)

    fig.tight_layout()
    out = OUT_DIR / "fig3_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── 英語版 図1: Quadrant ────────────────────────────────────────────────
def plot_quadrant_en():
    lat_max, lat_min = max(LATENCY), min(LATENCY)
    speed_norm = [(lat_max - l) / (lat_max - lat_min) for l in LATENCY]
    acc_norm   = [(s - 2.0) / 1.0 for s in SCORES]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.axhspan(0.5, 1.05, xmin=0.5, xmax=1.05, color="#d9f0cb", alpha=0.4, zorder=0)
    ax.axhspan(-0.05, 0.5, xmin=-0.05, xmax=0.5, color="#fce4d6", alpha=0.3, zorder=0)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axvline(0.5, color="gray", lw=0.8, ls="--")

    labels_en = ["Case A\n20B Plain", "Case B\n8B LoRA FT", "Case C\n20B GraphRAG"]
    for i, (sx, sy, label, color) in enumerate(
            zip(speed_norm, acc_norm, labels_en, COLORS)):
        ax.scatter(sx, sy, s=300, color=color, zorder=5, edgecolors="white", lw=1.5)
        offset = [(0.04, 0.04), (-0.20, 0.04), (0.04, -0.08)][i]
        ax.annotate(
            label, (sx, sy),
            xytext=(sx + offset[0], sy + offset[1]),
            fontsize=10, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85)
        )

    ax.text(0.75, 0.95, "Ideal Zone", ha="center", va="top",
            fontsize=9, color="#2e7d32", style="italic")
    ax.text(0.25, 0.95, "Accurate\nbut Slow", ha="center", va="top",
            fontsize=9, color="#555", style="italic")
    ax.text(0.75, 0.05, "Fast but\nLess Accurate", ha="center", va="bottom",
            fontsize=9, color="#555", style="italic")
    ax.text(0.25, 0.05, "Needs\nImprovement", ha="center", va="bottom",
            fontsize=9, color="#c62828", style="italic")

    ax.set_xlabel("Inference Speed  (slow ← → fast)", fontsize=11)
    ax.set_ylabel("Answer Quality   Judge avg /3", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Slow\n(42 s)", "Mid\n(28 s)", "Fast\n(14 s)"], fontsize=9)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["2.00", "2.50", "3.00"], fontsize=9)
    ax.set_title("① Approach Trade-off: Inference Speed vs Answer Quality",
                 fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    out = OUT_DIR / "fig1_quadrant_en.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── 英語版 図2: Judge Score Bar ─────────────────────────────────────────
def plot_scores_en():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [1, 1.6]})

    short = ["Case A\n20B Plain", "Case B\n8B LoRA FT", "Case C\n20B GraphRAG"]
    bars = ax1.bar(short, SCORES, color=COLORS, edgecolor="white",
                   linewidth=1.5, width=0.5, zorder=3)
    ax1.set_ylim(0, 3.5)
    ax1.set_ylabel("Judge Average Score  (/3)", fontsize=11)
    ax1.set_title("Average Score", fontsize=11, pad=8)
    ax1.axhline(3.0, color="gray", lw=0.8, ls="--")
    ax1.tick_params(labelsize=9)
    ax1.set_yticks([0, 1, 2, 2.29, 2.62, 2.92, 3])
    ax1.set_yticklabels(["0", "1", "2", "2.29", "2.62", "2.92", "3.0"], fontsize=8)
    ax1.grid(axis="y", zorder=0, alpha=0.4)
    for bar, score in zip(bars, SCORES):
        ax1.text(bar.get_x() + bar.get_width() / 2, score + 0.05,
                 f"{score:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.annotate("", xy=(1, SCORES[1]), xytext=(0, SCORES[0]),
                 arrowprops=dict(arrowstyle="->, head_width=0.2",
                                 color="#c62828", lw=1.5))
    ax1.text(0.5, (SCORES[0] + SCORES[1]) / 2 + 0.06, "+0.63",
             ha="center", color="#c62828", fontsize=9, fontweight="bold")

    score_labels = ["0 pt", "1 pt", "2 pt", "3 pt"]
    dist_colors  = ["#c62828", "#f4a261", "#2196f3", "#4caf50"]
    cases_short  = ["Case A", "Case B", "Case C"]
    x = np.arange(len(cases_short))
    bottom = np.zeros(3)
    for j, (lbl, col) in enumerate(zip(score_labels, dist_colors)):
        vals = [DIST[k][j] for k in DIST]
        ax2.bar(x, vals, bottom=bottom, label=lbl, color=col,
                edgecolor="white", linewidth=0.8, zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax2.text(xi, b + v / 2, f"{v}%", ha="center", va="center",
                         fontsize=9, fontweight="bold", color="white")
        bottom += np.array(vals)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Case A\n20B Plain", "Case B\n8B LoRA FT",
                          "Case C\n20B GraphRAG"], fontsize=9)
    ax2.set_ylabel("Questions (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("Score Distribution (0–3 pt)", fontsize=11, pad=8)
    ax2.legend(loc="upper right", fontsize=9, ncol=2)
    ax2.grid(axis="y", zorder=0, alpha=0.4)
    ax2.tick_params(labelsize=9)

    fig.suptitle("② Judge Score Comparison  (LLM-as-Judge: Qwen2.5:14B / 100 questions)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig2_scores_en.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── 英語版 図3: Evolution Timeline ──────────────────────────────────────
def plot_evolution_en():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")

    BOX_HALF = 0.22
    MARGIN   = 0.06
    centers  = [0.0, 1.1, 2.2]

    stages_en = [
        ("Case A\nBaseline",
         "GPT-OSS Swallow 20B\nLatency: 42.2 s",
         "Judge avg\n2.29 / 3", "+0.00", COLORS[0], centers[0]),
        ("Case C\nKnowledge Graph",
         "GPT-OSS Swallow 20B\n+ Neo4j GraphRAG\nLatency: 31.1 s",
         "Judge avg\n2.62 / 3", "+0.33", COLORS[2], centers[1]),
        ("Case B\nLoRA FT  (Best)",
         "Swallow-8B QLoRA FT\nn=715 pairs\nLatency: 14.2 s",
         "Judge avg\n2.92 / 3", "+0.63", COLORS[1], centers[2]),
    ]

    # ① Draw boxes first
    for label, detail, score, delta, color, xc in stages_en:
        box = mpatches.FancyBboxPatch(
            (xc - BOX_HALF, 0.0), BOX_HALF * 2, 1.0,
            boxstyle="round,pad=0.04", linewidth=2,
            edgecolor=color, facecolor=color + "20", zorder=2
        )
        ax.add_patch(box)
        ax.text(xc, 0.9, label, ha="center", va="top",
                fontsize=10, fontweight="bold", color=color, zorder=3)
        ax.text(xc, 0.6, detail, ha="center", va="top",
                fontsize=8, color="#333", linespacing=1.5, zorder=3)
        ax.text(xc, 0.16, score, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, zorder=3)
        badge_color = "#4caf50" if delta != "+0.00" else "#aaa"
        ax.text(xc + 0.15, 1.05, delta, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none"))

    # ② Arrows between boxes
    for i in range(len(centers) - 1):
        x_start = centers[i]     + BOX_HALF + MARGIN
        x_end   = centers[i + 1] - BOX_HALF - MARGIN
        ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5),
                    arrowprops=dict(arrowstyle="->, head_width=0.12",
                                   color="gray", lw=2),
                    zorder=5)

    ax.set_title("④ Evolution of Approaches and Accuracy Improvement",
                 fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    out = OUT_DIR / "fig3_evolution_en.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


# ── メイン ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 日本語版
    plot_quadrant()
    plot_scores()
    plot_evolution()
    # 英語版
    plot_quadrant_en()
    plot_scores_en()
    plot_evolution_en()
    print("完了 —", OUT_DIR)
