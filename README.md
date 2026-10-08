# Sparse-Offload Inference Engine

A from-scratch CUDA/Python LLM inference engine, built milestone by milestone
against [PROJECT_SPEC.md](PROJECT_SPEC.md): custom INT4/INT8 quantized GEMV
kernels, weight streaming over PCIe for models that don't fit in VRAM,
Dynamic Input Pruning (skip the MLP channels a token doesn't need), and a
GPU-resident hot-channel cache to avoid re-streaming the channels most tokens
need anyway.

**Status: all milestones M0-M9 done.** Every number below is backed by a
checked-in CSV in `reports/` (PROJECT_SPEC.md sec 6/9's own rule) -- see
[docs/RESULTS.md](docs/RESULTS.md) for the full per-milestone breakdown, and
[docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md) for the running log of what
actually happened, including five real, hardware-verification-only-visible
bugs found and fixed along the way.

## Result: input-dependent pruning loses to dense offload

**In this implementation, Dynamic Input Pruning (with or without the hot
cache) is slower than plain dense offload and much less accurate.** No
configuration measured beats dense on throughput.

On a single NVIDIA T4 (15 GB, pinned-host PCIe at ~12.3 GB/s measured),
decoding Qwen3-1.7B (group-128 symmetric INT4, weights offloaded to host
memory, batch 1, 70-token teacher-forced eval passage), at k/I = 0.5:

| mode | bytes/token saved | tok/s | perplexity |
|---|---|---|---|
| dense offload (same kernels) | 0% | **12.10** | 16.50 |
| plain DIP | 25.0% | 7.70 | 231.26 (14.0x) |
| cache-aware DIP, cache = 20% of I | 33.6% | 7.29 | 231.26 (14.0x) |

Source: `reports/m9_ablation_matrix.csv` (one seed). `reports/m8_ablation.csv`
is an earlier, independent run of the same three modes (cache = 10% of I)
and agrees: 12.08 / 7.64 / 7.03 tok/s, identical perplexities.

**Why it loses -- two separate problems:**

1. **The pruning mechanism costs more than the bytes it saves.** DIP at
   k/I = 1.0 keeps every channel and transfers exactly as many bytes as
   dense, yet runs at 5.58 tok/s vs. 12.10 -- the selection machinery alone
   adds ~97 ms to an ~83 ms token before any bytes are saved. Turning the
   cache on adds a further fixed cost: at k/I >= 0.375, going from no cache
   to a 5% cache *lowers* throughput by 0.7-1.1 tok/s, and at k/I >= 0.5
   even a 20% cache never wins it back. Nsight Compute (at Qwen3-14B shapes)
   puts ~96% of captured kernel time in three kernels:
   `topk_threshold_select` (runs on 1 of the T4's 40 SMs, ~144x over its
   target after a determinism fix) and the two down-projection accumulate
   kernels (half the SMs, poorly coalesced loads). Each decode layer also
   blocks on a device-to-host copy of the selected indices before it can
   start the row gather; the cache path adds a second sync for the miss
   count. See `reports/m9_ncu_summary.md`.
2. **Even with zero overhead, the trade wouldn't be worth it.** If per-token
   time scaled purely with bytes transferred (M6 shows every streamed matrix
   is transfer-bound), saving 33.6% of bytes caps the gain at ~1.5x. That
   ceiling costs a 14x perplexity increase, because top-k by raw gate
   magnitude is the only channel-selection criterion tried, and it throws
   away channels the model needs.

**What does hold up:** cache-aware DIP's perplexity is **bit-identical** to
plain DIP's (231.26173400878906 at k/I = 0.5, and matching at all 18
cache-size/k points of the M9 sweep, plus two independent M8 runs). The
cache changes only where weight bytes come from, never the arithmetic. It
also saves real bytes: 25.0% -> 33.6% fewer per token at k/I = 0.5.

## The three key plots

**Roofline** (M6): every weight matrix streamed over PCIe is transfer-bound,
not compute-bound -- the copy takes 5.5x-10.2x longer than the GEMV kernel
that consumes the same bytes once they land on GPU. This is why M6's whole
design (overlap the next weight's copy with the current one's compute)
matters more than GEMV kernel speed, and why M4's INT4-vs-fp16 kernel speed
gap turned out not to matter for the system as a whole.

![Roofline: every weight matrix is transfer-bound](reports/m6_roofline.png)

**Accuracy-vs-bytes-saved Pareto curve** (M7, corrected 2026-09-18 after a
real kernel non-determinism bug was found and fixed -- see
[docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md)): perplexity degrades
roughly log-linearly down to k/I=0.25, then falls off a real cliff at
k/I=0.125. Even the mildest pruning point tested (k/I=0.75, 12.5% of bytes
saved) already costs 3.5x perplexity; k/I=0.5 (14x) is the least-bad point
that saves a meaningful share of bytes, not a usable one.

![M7 Pareto curve](reports/m7_pareto.png)

**Ablation matrix** (M8's dense -> +DIP -> +cache-aware DIP table, extended
by M9's sweep over k/I x cache size -- full tables in
[docs/RESULTS.md](docs/RESULTS.md#m8----cache-aware-dip)). Each line starts
at plain DIP (cache = 0); the dashed line is dense offload. No point
reaches it. A cache beats plain DIP only at k/I <= 0.375, where perplexity
is already 42x dense or worse:

![M9: DIP and cache-aware DIP throughput vs. cache size, against dense](reports/m9_ablation_matrix.png)

## Caveats (read before citing any number above)

- **One model** (Qwen3-1.7B), **one 70-token eval passage**,
  top-k-by-raw-gate-magnitude as the only channel-selection criterion tried.
  Not load-bearing for a stronger claim without a longer eval or the 14B
  model -- see [docs/RESULTS.md](docs/RESULTS.md)'s M7 section.
- **`topk_threshold_select` (the top-k selection kernel) is ~144x over its
  own single-digit-microsecond target**, including an honest ~8x regression
  from a determinism fix made 2026-09-18 (necessary: the kernel previously
  produced different perplexity on back-to-back runs of the identical
  model). This is the clearest, best-quantified target for future kernel
  work -- see `reports/m9_ncu_summary.md` and
  [docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md).
- **Throughput numbers are single-seed and noisy.** Repeated M8 runs of the
  same cache-aware configuration spread over 7.03-7.31 tok/s, and the M9
  sweep is not monotonic in cache size at k/I = 0.25 (9.17 -> 8.15 -> 9.77
  tok/s). Differences under ~0.5 tok/s between nearby points shouldn't be
  read as real. The dense-vs-DIP gap (12.1 vs. <= 7.7 at k/I >= 0.5) is far
  larger than that noise. Perplexity and bytes/token are deterministic and
  reproduce exactly.
- **Measured at 1.7B, not the 14B model the spec targets.** At 14B the
  transfer share of each token is larger, which favors byte savings, but
  `topk_threshold_select` and the down-projection kernels also get slower
  with I. Which effect wins there is unmeasured.

## Reproducing this

This project's own dev machine has no NVIDIA GPU -- everything CUDA-dependent
runs on a Colab or Kaggle T4 session, never locally. A stranger with GPU
access reproduces the results above the same way this project's own
sessions do:

```
git clone <this repo>
cd sparse-offload-infer
make build             # compiles the CUDA extension (needs nvcc; on a
                        # machine with no CUDA toolkit, this falls back to a
                        # pure-Python install automatically -- see setup.py)
make test               # full test suite; GPU-requiring files self-skip on
                         # a CPU-only machine (this is what CI runs)
make bench                # M0-M7's core benchmark sweep (fast; no model
                           # download)
make bench-m7-pareto       # M7's Pareto curve (downloads Qwen3-1.7B,
                            # ~3.4GB; writes reports/m7_pareto.csv)
make bench-m8-ablation           # M8's single-point ablation table
make bench-m9-ablation-matrix    # M9's fuller cache-size sweep (needs
                                  # reports/m7_pareto.csv from the step above)
make profile-m9-kernels           # M9's Nsight Compute profile (needs ncu,
                                   # ships with the CUDA toolkit)
python bench/plot_results.py       # regenerates the 3 plots above from the
                                    # CSVs (pure matplotlib, no CUDA needed)
```

No local NVIDIA GPU? Use [notebooks/colab_bootstrap.ipynb](notebooks/colab_bootstrap.ipynb)
on a Colab/Kaggle T4 session -- the exact workflow this project itself used
for every number in `reports/`.

## More

- **New to this project? Start here:** [docs/OVERVIEW.md](docs/OVERVIEW.md) --
  what this is, why each piece exists, how it fits together, and the
  milestone-by-milestone story in plain language
- Full spec and milestone plan: [PROJECT_SPEC.md](PROJECT_SPEC.md)
- Complete results, every milestone: [docs/RESULTS.md](docs/RESULTS.md)
- Design decisions and deviations from spec: [docs/DESIGN.md](docs/DESIGN.md)
- Running learning log (including every bug found verifying this project on
  real hardware, and why "verified" needs to say at what scale):
  [docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md)
