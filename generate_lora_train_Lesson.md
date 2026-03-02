# LoRA 学習用データ生成における課題とレッスン

> 対象スクリプト: `scripts/03b_generate_lora_qa_graph.py`  
> 実行日時: 2026-03-02  
> 目標: 268リレーション × 3問 = **804問**  
> 実績: **715問**（欠損 89問）

---

## 1. エラー概要

| エラー種別 | 発生件数（概算） | 欠損への影響 | 優先度 |
|---|---|---|---|
| `Extra data` (JSON パースエラー) | 約 30 件 | リトライで一部救済、失敗分は欠損 | ★★★ 高 |
| `Read timed out` (タイムアウト) | 約 15 件 | 全リトライ失敗でスキップ（主因） | ★★★ 高 |
| 制御文字 / trailing comma | 数件 | 一部リトライで救済 | ★★ 中 |
| 空配列 `[]` 返却 | 数件 | スキップ | ★ 低 |
| 類似度超過（独立性フィルタ） | 1 件 | 正常除外（設計通り） | — |

---

## 2. エラー詳細と原因

### 2-1. `Extra data: line N column M`（最多発）

**エラーログ例:**

```
[WARN] JSON パースエラー (attempt 1): Extra data: line 4 column 2 (char 356)
[WARN] JSON パースエラー (attempt 2): Extra data: line 4 column 2 (char 340)
```

**原因:**  
`re.search(r"\[.+\]", content, re.DOTALL)` のグリーディマッチが、
JSON 配列の正しい閉じ `]` を超えて、本文中に現れる別の `]`（例: `[河川法 第X条]`、
`[注]` など）まで取り込んでしまう。
`json.loads()` は最初の有効な配列を解析した後に余剰文字を検知し `Extra data` エラーを送出する。

**現行コードの問題箇所:**

```python
m = re.search(r"\[.+\]", content, re.DOTALL)
if m:
    parsed = json.loads(m.group())   # ← Extra data が発生
```

**改善策:**  
`json.JSONDecoder().raw_decode()` を用い、`[` の出現位置から順に試行して
配列の終端を正確に特定する（余剰文字を無視できる）。

```python
def extract_json_array(content: str) -> list:
    cleaned = re.sub(r',\s*([}\]])', r'\1', content)          # trailing comma 除去
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)  # 制御文字除去

    for m in re.finditer(r'\[', cleaned):
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned, m.start())
            if isinstance(obj, list) and obj:
                return obj
        except json.JSONDecodeError:
            continue

    # 単一オブジェクトフォールバック
    for m in re.finditer(r'\{', cleaned):
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned, m.start())
            if isinstance(obj, dict) and 'question' in obj:
                return [obj]
        except json.JSONDecodeError:
            continue

    return []
```

---

### 2-2. `Read timed out` (タイムアウト)

**エラーログ例:**

```
[ERROR] Ollama エラー (attempt 1): HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)
[ERROR] Ollama エラー (attempt 2): HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)
[ERROR] Ollama エラー (attempt 3): HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)
```

**原因:**  
`num_predict=-1`（無制限）を設定したため、まれにモデルが過剰に長い応答を生成し
120 秒のタイムアウト上限を超える。
3 回分のリトライすべてが失敗したリレーションは丸ごとスキップされ、
最大 3 問の欠損が生じる（89 問欠損の主因）。

**改善策:**
- `num_predict` を適切な上限に設定する。  
  3 問 × 350 字 ≒ 1,050 文字 → **`num_predict=2048`** で十分かつ安全。
- タイムアウト後のリトライでは `num_predict` をさらに小さく（例: 1024）にして
  短縮版を取得するフォールバックを追加する。

```python
# 改善後のオプション例
"options": {
    "temperature":    0.7,
    "repeat_penalty": 1.2,
    "num_ctx":        4096,
    "num_predict":    2048,   # ← -1（無制限）から変更
    "stop":           ["<|end|>", "<|start|>"],
},
```

---

### 2-3. 制御文字 / trailing comma

**エラーログ例:**

```
[WARN] JSON パースエラー (attempt 1): Invalid control character at: line 3 column 349 (char 412)
[WARN] JSON パースエラー (attempt 1): Expecting ',' delimiter: line 7 column 430 (char 1222)
```

**原因:**  
LLM が JSON 文字列内にエスケープされていない生の制御文字（`\r`、バックスペース等）を
出力する場合がある。また、配列末尾に trailing comma `},]` を付ける誤った JSON を
生成することがある（Python 標準の `json.loads` はいずれも非対応）。

**改善策:**  
パース前処理として:
1. `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` の制御文字を除去（改行・タブは保持）
2. `,\s*([}\]])` → `\1` で trailing comma を除去

上記は前項の `extract_json_array` 関数内に含まれる。

---

### 2-4. 空配列 `[]` 返却

**エラーログ例:**

```
[WARN] JSON パースエラー (attempt 1): JSON 配列なし: [
```

**原因:**  
LLM が `[]` や `[ ]` のみを返し、問題・解答を生成しないケースがある。
プロンプトの指示が抽象的すぎる場合や、リレーション文脈が短すぎて
LLM が適切な問題を生成できない場合に発生。

**改善策:**
- 空リスト検知時は即リトライし、プロンプトに「**必ず{n}問を JSON 配列形式で出力してください。空配列は不可です**」を追加する。
- description が空のノードに対しては文脈が薄くなるため、`--n_per_rel 1` に削減するか、テンプレートを補強する。

---

## 3. 設計上の課題

### 3-1. `RETRY` 定数の命名と動作の不整合

現行コードでは `RETRY = 2` と設定しているが、ループは `range(RETRY + 1)` のため
実際は **3 回試行**する（初回 + 2 リトライ）。
変数名を `MAX_ATTEMPTS = 3` に変更して意図を明確にする。

### 3-2. チェックポイントが 10 リレーション粒度

タイムアウトが連続して 9 問失敗した直後にクラッシュすると、
最大 9 リレーション分（最大 27 問）が失われる。
チェックポイント間隔を **5 リレーション** に短縮するか、
1 リレーション完了ごとに追記モードで書き込む方式（`mode="a"`）が望ましい。

### 3-3. 独立性フィルタのしきい値設定

`SIMILARITY_THRESH = 0.45` であり、今回は 1 問のみ除外で済んだ。
ただし文字 bigram Jaccard はドメイン特有の専門用語が多いと類似度が
過大評価されるリスクがある。評価後 FP/FN を確認し、必要に応じて
**意味埋め込み（sentence-transformers）** による類似度フィルタへ切り替えを検討する。

### 3-4. リレーション種別の偏り

生成実績（715 問）の種別内訳:

| rel_type | 問数 | 割合 |
|---|---|---|
| HAS_CHAPTER | 212 | 29.7% |
| DESCRIBED_IN | 89 | 12.4% |
| HAS_SECTION | 86 | 12.0% |
| HAS_ITEM | 63 | 8.8% |
| REQUIRES | 56 | 7.8% |
| SUBJECT_TO | 55 | 7.7% |
| DEFINED_IN | 54 | 7.6% |
| USED_IN | 44 | 6.2% |
| MITIGATES | 28 | 3.9% |
| AFFECTS | 21 | 2.9% |
| PRECEDES | 7 | 1.0% |

`HAS_CHAPTER` が全体の 30% を占め、`PRECEDES` は 7 問に留まる。
LoRA 学習時のデータ不均衡を緩和するため、少数クラスに対しては
`--n_per_rel` を増やす（例: `PRECEDES` は 5〜6 問）、あるいは重み付きサンプリングが有効。

---

## 4. 改善後スクリプトの変更点まとめ

| 変更箇所 | 変更前 | 変更後 |
|---|---|---|
| JSON 抽出 | `re.search(r"\[.+\]", ..., DOTALL)` | `raw_decode()` による正確な終端検出 |
| JSON 前処理 | なし | 制御文字除去 + trailing comma 除去 |
| `num_predict` | `-1`（無制限） | `2048`（適切な上限） |
| リトライ命名 | `RETRY = 2` | `MAX_ATTEMPTS = 3` に変更 |
| チェックポイント間隔 | 10 リレーション毎 | 5 リレーション毎（または追記モード） |

---

## 5. 次回生成時の推奨手順

```powershell
# 1. スクリプト修正後に dry_run で文脈確認
python scripts/03b_generate_lora_qa_graph.py --dry_run

# 2. 少数クラスを優先して不足分補完（例: PRECEDES を追加生成）
python scripts/03b_generate_lora_qa_graph.py --start 1 --out data/lora/train_graph_rels_v2.jsonl

# 3. 生成後のサマリー確認
python scripts/_show_lora.py
```

---

*作成: 2026-03-02*
