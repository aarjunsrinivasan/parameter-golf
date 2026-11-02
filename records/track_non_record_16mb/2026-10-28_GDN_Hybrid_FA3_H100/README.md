# GDN Hybrid at 32k on H100: FlashAttention-3 and an aligned head dim

**Track:** Non-Record Submission (Unlimited Compute)
**Base record:** [2026-04-04_GDN_Hybrid_LongContext](../2026-04-04_GDN_Hybrid_LongContext/)
**Hardware:** 1× H100 80GB HBM3, 600 seconds per run
**Date:** 2026-10-28

---

## Summary

The base record found that a Gated DeltaNet hybrid (7 GDN + 2 attention layers) beats a
full-attention baseline at 32k context by **−0.235** bpb, with both arms using PyTorch SDPA
(FlashAttention-2). This follow-up asks how much of that margin survives a fair attention kernel.
Both arms now use **FlashAttention-3**, and the schedule is a fraction of the wall-clock budget, so
neither arm is mis-tuned for its step count.

- **FA3 removes about 70% of the margin.** With both arms on FA3 the hybrid still wins, by
  **−0.065**. Full attention is the arm FA3 helps most (9/9 layers vs 2/9), at 1.76× vs 1.23×.
- **Aligning the GDN head dim to 64 widens the margin again.** It goes to **−0.092**, with
  5 heads × 64 instead of 8 × 42 and fewer parameters than the baseline. The gain is pure speed
  (15% faster steps); per-step loss is identical.
- **Kimi Delta Attention does not beat GDN.** Swapping GDN's per-head decay for KDA's per-channel
  decay costs 14–23% step time and buys no per-step quality. KDA trails GDN at both shapes
  (+0.045 at 8×42, +0.021 at 5×64).

All numbers are 2-seed means of `final_int8_zlib_roundtrip` val_bpb, run on the same machine.

| hybrid − baseline (val_bpb, seq 32768) | |
|---|---|
| SDPA / FA2, base record | −0.235 |
| FA3, GDN 8×42 (base record's shape) | **−0.065** |
| FA3, GDN 5×64 | **−0.092** |

---

## Results

### FA3 matrix (seq 32768, 2 seeds each)

| Arm | params | ms/step | steps | val_bpb per seed | mean ± half-range | Δ vs baseline | max bytes |
|---|---|---|---|---|---|---|---|
| baseline (full attention) | 17,059,912 | 1253 | 480/479 | 1.4816 (s1339), 1.4843 (s42) | 1.4829 ± 0.0013 | — | 13,479,205 |
| GDN hybrid 8×42 | 17,424,332 | 1062 | 565/566 | 1.4182 (s1342), 1.4187 (s42) | 1.4184 ± 0.0003 | **−0.0645** | 14,606,470 |
| **GDN hybrid 5×64** | 17,002,582 | **903** | 661/668 | 1.3902 (s1342), 1.3913 (s42) | **1.3908 ± 0.0005** | **−0.0922** | 14,547,635 |
| KDA hybrid 8×42 | 17,862,532 | 1304 | 460/461 | 1.4639 (s1342), 1.4625 (s42) | 1.4632 ± 0.0007 | −0.0197 | 14,965,058 |
| KDA hybrid 5×64 | 17,677,235 | 1034 | 582/580 | 1.4093 (s1342), 1.4144 (s42) | 1.4119 ± 0.0026 | −0.0711 | 15,522,344 |

`max bytes` is the int8+zlib model plus code. All are under 16,000,000.

### Where the base record's margin went

| Arm | SDPA ms/step | FA3 ms/step | FA3 speedup |
|---|---|---|---|
| baseline | 2202 (base record) | 1253 | **1.76×** |
| GDN hybrid 8×42 | 1308 (gate run, this machine) | 1062 | 1.23× |

The hybrid's throughput lead at 32k falls from 1.62× (SDPA) to **1.18×** (FA3). At 5×64 it
recovers to **1.39×**. The hybrid's step-time advantage was the main reason it won, so a faster
attention kernel for the baseline narrows the gap directly.

### Why 5×64 is faster

With `MODEL_DIM=448` and 8 heads, `head_dim = round(56 × 0.75) = 42`, which the FLA kernels pad to
64. `GDN_N_HEADS=5 GDN_HEAD_DIM_RATIO=0.72` gives K=64, V=128 at almost the same key/value width
(320/640 vs 336/672). Parameter count drops slightly, and the GDN layers stop wasting a third of
their key lanes.

In-training val_bpb (2-seed mean) shows the per-step quality is the same:

| step | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| baseline | 2.0905 | 1.7404 | 1.5895 | 1.5117 | — |
| GDN 8×42 | 1.9013 | 1.6557 | 1.5364 | 1.4766 | 1.4324 |
| GDN 5×64 | 1.8976 | 1.6399 | 1.5296 | 1.4760 | 1.4325 |

So 5×64 wins by fitting about 100 more steps into 600 s. The larger recurrent state
(5·64·128 vs 8·42·84 per layer, +45%) buys nothing measurable at this scale. (Warmdown starts at
64% of wall-clock, so it starts at different steps per arm; checkpoints at and before step 300 are
pre-warmdown for all three.)

### Kimi Delta Attention ablation

[Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) uses the same 3:1 linear/global hybrid
pattern, with KDA in place of GDN. `GDN_MIXER=kda` is an **isolated ablation**: only the decay gate
changes. Everything else in `GDNLayer` stays the same: beta×2, SiLU output gate, short convs, init.

| | GDN | KDA (`GDN_MIXER=kda`) |
|---|---|---|
| decay gate | one scalar **per head**: `w_a: dim→H`, `g:[B,T,H]` | one value **per key channel**: low-rank `dim→head_v_dim→key_dim`, `g:[B,T,H,K]` |
| `dt_bias` | `[H]` | `[H·K]` |
| kernel | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | `fla.ops.kda.chunk_kda` |

`bench/check_kda.py` (run by `setup_runpod.sh`) checks that `chunk_kda` with a gate that is constant
over K reproduces `chunk_gated_delta_rule` (rel. error 5.5e-3, bf16).

| Mixer, shape | params | ms/step | steps | val_bpb (2-seed) | vs GDN same shape |
|---|---|---|---|---|---|
| GDN 8×42 | 17,424,332 | 1062 | 565 | 1.4184 | — |
| KDA 8×42 | 17,862,532 | 1304 (+23%) | 460 | 1.4632 | **+0.045** |
| GDN 5×64 | 17,002,582 | 903 | 665 | 1.3908 | — |
| KDA 5×64 | 17,677,235 | 1034 (+14%) | 581 | 1.4119 | **+0.021** |

In-training val_bpb, 2-seed mean, before any arm's warmdown (KDA 8×42 enters warmdown near step 295):

| step | 100 | 200 | 300 |
|---|---|---|---|
| GDN 8×42 | 1.9013 | 1.6557 | 1.5364 |
| KDA 8×42 | 1.9034 | 1.6833 | 1.5513 |
| GDN 5×64 | 1.8976 | 1.6399 | 1.5296 |
| KDA 5×64 | 1.8918 | 1.6740 | 1.5397 |

- **Per-channel decay buys nothing per step at this scale.** KDA is level with GDN or slightly
  behind at matched steps, at both shapes.
- **It costs steps.** KDA's kernel is 23% slower at 8×42 (the unaligned 42 pads to 64 in the
  per-channel gate path too) and 14% slower at 5×64, so it gets 85–105 fewer steps in 600 s.
- Alignment helps KDA even more than GDN (−0.051 vs −0.028), but not enough to catch up.
- KDA 5×64 has the largest seed spread here (±0.0026), still well below the +0.021 gap.

### Reproduction gate

Before the FA3 runs, the base record's hybrid was rerun unchanged (SDPA, step-count schedule 79/158,
seed 1342) in this environment:

| | base record (H100) | this machine (H100) |
|---|---|---|
| params | 17,424,332 | 17,424,332 |
| train_loss step 1 / 10 / 100 | 6.9344 / 5.9574 / 3.2059 | 6.9344 / 5.9570 / 3.2037 |
| ms/step | 1361 | 1308 |
| steps in 600 s | 440 | 459 |
| val_bpb | 1.4703 | **1.4630** |

The trajectory reproduces. The score is slightly better because this H100 is 4% faster per step,
and the base scripts' warmdown is tied to step count × step time
(`logs_experiment/gate_sdpa_h100_seq32768_s1342.txt`).

---

## What changed vs the base record

| | base record | this folder |
|---|---|---|
| attention kernel | SDPA (FlashAttention-2) | FA3 from `kernels-community/flash-attn3` (`ATTN_BACKEND=fa3`) |
| schedule | `MUON_MOMENTUM_WARMUP_STEPS` / `WARMDOWN_ITERS` hand-set from measured step times | `MUON_MOMENTUM_WARMUP_FRAC=0.18`, `WARMDOWN_FRAC=0.36` of the 600 s budget |
| GDN shape | 8 heads × 42 | 8 × 42 and 5 × 64 (`GDN_N_HEADS`, `GDN_HEAD_DIM_RATIO`) |
| environment | torch 2.9.1+cu128, fla unpinned | torch 2.9.1+cu128, fla 0.4.2, kernels 0.16.0 (pinned) |

The fraction schedule keeps the base record's 0.18 / 0.36 ratios, but those ratios now follow
whatever step time the kernel produces. Without it, a faster kernel silently shrinks warmup and
warmdown, which hurts the baseline most because FA3 speeds it up most.

Everything else (optimizer, LRs, batch 524,288 tokens, model widths, int8+zlib evaluation on the
full FineWeb val split) is unchanged. With defaults (`ATTN_BACKEND=sdpa`, no `*_FRAC`), both scripts
reproduce the base record.

The FA3 swap, the 5×64 shape and the fraction schedule came out of earlier exploratory runs on
H200. This folder reruns the headline arms on H100, the challenge's target hardware.

`train_gpt_gdn.py` also has `GDN_COMPILE_MODE` / `GDN_FULLGRAPH` flags (bypassing FLA's
`torch.compiler.disable` wrapper). They were not used in the runs reported here.

---

## Reproducing

From the repo root, on a Hopper GPU:

```bash
R=records/track_non_record_16mb/2026-10-28_GDN_Hybrid_FA3_H100
bash $R/setup_runpod.sh     # uv venv: Python 3.12, torch 2.9.1+cu128, fla 0.4.2, kernels 0.16.0,
                            # 8 data shards, FA3 load check, KDA gate (bench/check_kda.py)
CONFIGS="baseline gdn_h8k42 gdn_h5k64 kda_h8k42 kda_h5k64" SEEDS="default 42" bash $R/run_core.sh
.venv/bin/python $R/summarize.py logs
```

`run_core.sh` runs one 1-GPU job per visible GPU, skips runs that already have a final score, and
writes `logs/core_<config>_<gpu>_seq<L>_s<seed>.txt`. On one H100 each run takes about 15–22 min
wall-clock. Only the 600 s of training steps are capped. Compile, the 20 warmup steps, the
full-val checks every 50 steps and the final int8 eval are not counted. `VAL_LOSS_EVERY=0` skips
the in-training val checks without changing the final score.

The reproduction gate:

```bash
PATH=$PWD/.venv/bin:$PATH ATTN_BACKEND=sdpa SEED=1342 RUN_ID=gate_sdpa_h100_seq32768_s1342 \
  bash $R/run_hybrid_long_context_single_h100.sh 32768
```

## Files

| Path | What |
|---|---|
| `train_gpt.py` | Full-attention baseline + `ATTN_BACKEND`, `*_FRAC` schedule |
| `train_gpt_gdn.py` | GDN hybrid + `ATTN_BACKEND`, `*_FRAC`, `GDN_N_HEADS`, `GDN_MIXER`, `GDN_COMPILE_MODE` |
| `run_{baseline,hybrid}_long_context_single_h100.sh` | As base record, with schedule settings overridable from the environment |
| `setup_runpod.sh`, `run_core.sh`, `summarize.py` | Environment setup, job queue, results table |
| `requirements.txt` | Pinned dependencies (torch installed separately with `--torch-backend=cu128`) |
| `logs_experiment/` | Full logs: 10 FA3 runs (5 configs × 2 seeds) + the SDPA gate |
| `../../../bench/check_kda.py` | KDA-vs-GDN numerical gate run by `setup_runpod.sh` |

## Limitations

- **Only seq 32768.** FA3 helps full attention most at long context, so the 8k–16k crossover in the
  base record likely moves right. That is untested here.
- Two seeds per arm. Seed spread is ≤0.0026, far below the gaps reported.
- KDA was tested only as a decay-gate ablation. The full Kimi-Linear layer (sigmoid low-rank output
  gate, unscaled beta) and its global layers (MLA, NoPE) were not.
- The 5×64 shape changes kernel alignment and recurrent-state size together. The per-step curves
  suggest state size doesn't matter here, but the two weren't separated.
- The FA3 Δs come from runs on the same machine and environment. The SDPA −0.235 is the base
  record's own number (different day, same GPU type, step-count schedule).
