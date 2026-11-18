# data/bench — data files of the WSP-Lang `dist-shift` suite (downloaded 2026-11-18, approved by the driver)

| file | source | size | sha256 (first 16) | licence | sanity |
|---|---|---|---|---|---|
| `hellaswag_val_full.txt` | https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt (the file llama.cpp's `scripts/get-hellaswag.sh` fetches) | 7,770,677 B | `d572539320eb2050` | HellaSwag: MIT (Zellers et al. 2019) | 60,252 lines; llama-perplexity loads 10,042 tasks |
| `winogrande-debiased-eval.csv` | https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/main/winogrande-debiased-eval.csv (llama.cpp's `scripts/get-winogrande.sh`) | 155,325 B | `6726173ef65ffdc4` | Winogrande: CC BY (Sakaguchi et al. 2019) | 1,268 lines; 1,267 tasks loaded |
| `mmlu-validation.bin` | https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/mmlu-validation.bin | 762,276 B | `470af3a74eccacfa` | MMLU: MIT (Hendrycks et al. 2021) | header uint32 1548 tasks + positions; must be passed with `-bf` (binary): text-mode `-f` truncates it on Windows ("failed to read task positions") |
| `wikitext2_test.txt` | https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip (the zip llama.cpp's `scripts/get-wikitext-2.sh` fetches; `wikitext-2-raw/wiki.test.raw` extracted with the standard library, 2026-11-18 07:40) | 1,290,590 B | `173c87a53759e020` | WikiText-2: CC BY-SA 3.0 | 4,358 lines; starts with the Robert Boulter article |
| `wikitext2_test.parquet` | https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet | 732,610 B | `5f1bea067869d048` | CC BY-SA 3.0 | not used (no parquet reader installed); kept only as provenance of the first attempt |

`logits/` holds the base models' saved logits (`<base_sha16>.bin`, written once per base by `--save-all-logits`, ≈ 247 KB
per token for a 248k vocabulary: 16 chunks × 512 tokens ≈ 2.0 GB per base). Git-ignored; regenerable.

Determinism checked on the CPU with the 0.8B (2026-11-18): HellaSwag task selection with `--hellaswag-tasks 8` gives
the same 8 tasks and the same accuracies on two runs, and does not change with `-s 7` (the selection uses the tool's
own fixed RNG, not the seed flag); the same code path is used for Winogrande and multiple choice.
