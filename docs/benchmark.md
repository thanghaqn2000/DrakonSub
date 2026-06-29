# Multi-sample benchmark & regression gate

DrakonSub dùng benchmark 4 mẫu ngắn để đo chất lượng pipeline dịch phụ đề (EN→VI) trước khi merge.

## Hai chế độ

| Mode | Mục đích | Dịch raw |
|------|----------|----------|
| `pipeline_regression` | **Hard gate** — đo chất lượng pipeline trên bản dịch raw cố định | Reuse job/debug raw + raw cache (`tests/fixtures/benchmark_raw` hoặc `artifacts/.../raw_cache`) |
| `end_to_end` | Theo dõi variance dịch end-to-end | Luôn `--fresh` (có thể `--cache-raw` để lưu cache) |

Manifest mẫu: [`scripts/benchmark_samples.json`](../scripts/benchmark_samples.json).

## Lệnh chuẩn (local)

### Pipeline regression (bắt buộc trước merge)

```bash
python scripts/run_multi_sample_benchmark.py \
  --engine openai \
  --use-raw-cache \
  --mode pipeline_regression

python scripts/ci_regression_check.py \
  --report artifacts/multi_sample_benchmark/pipeline_regression/benchmark_report.json
```

`--mode` là alias của `--benchmark-mode`.

### End-to-end (không block PR)

```bash
python scripts/run_multi_sample_benchmark.py \
  --engine openai \
  --fresh \
  --cache-raw \
  --mode end_to_end

python scripts/ci_regression_check.py --soft \
  --report artifacts/multi_sample_benchmark/end_to_end/benchmark_report.json
```

### Cả hai mode

```bash
python scripts/run_multi_sample_benchmark.py --engine openai --both-modes --use-raw-cache
```

## Guardrails (`ci_regression_check.py`)

**Pipeline regression (mặc định):**

- Contract pass: **4/4**
- `quality_score_min` ≥ **70**, `quality_score_avg` ≥ **80**
- Per-sample: `raise_price_17` ≥70, `no_rush_19` ≥70, `buffett_bitcoin_29` ≥80, `outsider_36` ≥75
- Tổng `semantic_alignment_errors` ≤ **3**
- `post_final_repair_text_lock_status` = pass (mọi sample)
- `benchmark_engine_status` = pass

**End-to-end (`--soft`):** ngưỡng thấp hơn — chỉ báo cáo, không chặn merge.

## Quy tắc merge

1. Chạy **pipeline regression** + `ci_regression_check` (exit 0).
2. Không commit `artifacts/multi_sample_benchmark/` hay `debug/` (đã `.gitignore`).
3. End-to-end chạy khi đổi logic dịch raw hoặc prompt translate; kết quả ghi trong PR nếu có biến động lớn.
4. Cập nhật `tests/fixtures/benchmark_raw/` khi đổi prompt/model raw có chủ đích (sau khi chạy `--cache-raw`).

## CI (GitHub Actions)

Workflow [`.github/workflows/benchmark-regression.yml`](../.github/workflows/benchmark-regression.yml):

- **PR:** smoke — chạy `ci_regression_check` trên golden report trong `tests/fixtures/golden/` (đảm bảo script + ngưỡng hợp lệ).
- **`workflow_dispatch`:** chạy benchmark đầy đủ nếu có secret `OPENAI_API_KEY` và job fixtures local (xem workflow).

**Hạn chế:** Job DrakonSub (`source.srt`) mặc định trỏ `jobs_root` máy dev (`/var/folders/.../drakonsub_jobs`). CI không có fixtures job đầy đủ → regression gate **đầy đủ chạy local** trước merge. Raw cache fixture (~32KB) nằm tại `tests/fixtures/benchmark_raw/` cho CI tương lai.

## Biến môi trường

- `TRANSLATION_ENGINE` trong `.env` có thể khác engine benchmark — luôn truyền `--engine openai` (hoặc `gemini`) explicit.
- Benchmark runner set `TRANSLATION_ENGINE` theo `--engine` khi chạy pass.
