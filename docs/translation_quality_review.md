# Translation Quality Review (Human Review Harness v1)

Xuất artifact cue-level để SA chấm điểm human 8/10 trước khi sửa pipeline.

## Command

```bash
# End-to-end (chấm chất lượng sản phẩm thật)
python scripts/export_translation_quality_review.py \
  --engine openai \
  --mode end_to_end \
  --samples raise_price_17,no_rush_19,buffett_bitcoin_29,outsider_36

# Pipeline regression (so sánh post-edit trên cached raw)
python scripts/export_translation_quality_review.py \
  --engine openai \
  --mode pipeline_regression \
  --use-raw-cache \
  --samples raise_price_17,no_rush_19,buffett_bitcoin_29,outsider_36

# Cả hai mode
python scripts/export_translation_quality_review.py --engine openai --both-modes --use-raw-cache
```

## Output

```
artifacts/translation_quality_review/
  {mode}/{sample_id}/review.csv
  {mode}/{sample_id}/review.md
  {mode}/{sample_id}/review.json
  quality_review_index.json
```

Các cột `SA_*` để trống cho SA điền. `recommended_fix_layer` có gợi ý heuristic (có thể sửa).

Artifacts không commit (local only).
