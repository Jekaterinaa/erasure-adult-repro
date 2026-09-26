# Reproducing the ERASURE Adult benchmark on a DGX Spark

I wanted to know whether the published Adult results from the ERASURE machine unlearning benchmark hold up on hardware very different from the authors'. I ran the benchmark's own config on an NVIDIA DGX Spark, a GB10 Grace Blackwell machine with an ARM (aarch64) CPU and CUDA 13. This repo has the scripts I used, the raw output, the logs and a comparison with the paper.

[ERASURE](https://github.com/aiim-research/ERASURE) is a modular framework for machine unlearning by D'Angelo et al. [1, 2]. The reference numbers I compare against come from their benchmark paper [3].

The short answer is yes. Utility and efficacy reproduce closely. Runtimes don't, which is expected on different hardware, but it matters more than I first thought because of how their LUMA metric is built.

## What I ran

- ERASURE at commit [`d5bbbe5`](https://github.com/aiim-research/ERASURE/tree/d5bbbe59365dec761d98bbc4427d2585d0dc1d63) (tagged "CIKM")
- Config `configs/benchmark/tabular/adult/fs_05/adult_1mlp_05%_seed0.jsonc`: a one-layer MLP on the UCI Adult dataset, 5% of the training set to forget, seed 0

## Data

The dataset isn't in this repo because it doesn't need to be. The benchmark config uses ERASURE's UCI data source, which downloads Adult (Becker & Kohavi, 1996, UCI Machine Learning Repository, dataset id 2) through the `ucimlrepo` package the first time it runs. The splits are made inside the config: 80/20 train/test, then 5% of the training set as the forget set and the rest as the retain set, all seeded with seed 0. Running the same config should give you the same splits.

## The run

One config is actually 38 runs. It trains the original model and a Gold model (retrained from scratch on the retain set only), then runs 12 unlearning methods at 3 settings each. The paper reports the best setting per method. The whole thing took 68 min 20 s.

## Getting it running on ARM

ERASURE installed cleanly on aarch64 with a single change: pip must not install `torch` or `torchvision` from `requirements.txt`. On this machine PyTorch has to come from the CUDA 13.0 wheel index, and letting pip resolve it can silently replace it with a CPU-only build. Nothing errors when that happens; training just gets about 100x slower. `scripts/install.sh` installs torch from the right index first and then filters those two lines out of the requirements.

I expected trouble from the two C extensions, `picologging` and `python_papi`, but both built from source without complaint.

```bash
git clone https://github.com/aiim-research/ERASURE.git ~/ERASURE
git -C ~/ERASURE checkout d5bbbe5

bash scripts/install.sh                  # creates ~/envs/erasure
bash scripts/run_adult.sh 2>&1 | tee adult.log
python scripts/summarize.py ~/ERASURE/dev/ICLR_benchmark/results/tabular/adult_1mlp_05%_seed_0.json
```

The paths and the torch index can be overridden with `ERASURE_DIR`, `VENV` and `TORCH_INDEX`.

The run segfaults at teardown (exit code 139). This happens after all 38 configurations have been evaluated and written, while ERASURE cleans its evaluator cache, so the results are complete. It does mean the exit code is useless as a success check, and I count the records in the results file instead.

## Results

The two reference models first, since everything else is measured against them:

| Model | Measure | Paper (3 seeds) | Mine (seed 0) | Difference |
|---|---|---|---|---|
| Original | F1 test | .790 ± .001 | .7938 | +.004 |
| Original | F1 forget | .799 ± .004 | .7976 | −.001 |
| Original | UMIA | .497 ± .002 | .5002 | +.003 |
| Gold | F1 test | .790 ± .002 | .7936 | +.004 |
| Gold | F1 forget | .797 ± .001 | .7931 | −.004 |
| Gold | UMIA | .499 ± .002 | .4964 | −.003 |

Every value is within about .004 of the published mean, which is the same size as the paper's own seed-to-seed spread.

The full table, produced by `scripts/summarize.py`. F1 is macro F1. UMIA is the unlearning membership inference attack accuracy, where .5 means the attacker can't tell forgotten samples from unseen ones. M_U and M_E are the utility and efficacy terms of the paper's LUMA metric, `exp(−3·‖x_Gold − x‖₁)` over (F1 test, F1 forget) and (UMIA) respectively, so both are 1.0 for Gold. The paper column shows their best setting per method.

| Method | Setting | F1 test | F1 forget | UMIA | M_U | M_E | Time (min) | Paper: F1 test / F1 forget / UMIA / time |
|---|---|---|---|---|---|---|---|---|
| Orig. | - | 0.7938 | 0.7976 | 0.5002 | 0.986 | 0.989 | 0.00 | 0.790 / 0.799 / 0.497 / 2.08 |
| Gold | lr=0.001 | 0.7936 | 0.7931 | 0.4964 | 1.000 | 1.000 | 1.28 | 0.790 / 0.797 / 0.499 / 2.08 |
| GD | lr=0.001 | 0.7926 | 0.7990 | 0.5003 | 0.980 | 0.988 | 0.67 | 0.790 / 0.802 / 0.498 / 1.12 |
|  | lr=0.0001 | 0.7913 | 0.8036 | 0.4977 | 0.962 | 0.996 | 0.66 |  |
|  | lr=1e-05 | 0.7905 | 0.8033 | 0.4982 | 0.961 | 0.995 | 0.66 |  |
| SRL | lr=0.001 | 0.7939 | 0.7973 | 0.4966 | 0.986 | 0.999 | 0.70 | 0.789 / 0.800 / 0.498 / 1.18 |
|  | lr=0.0001 | 0.7892 | 0.8001 | 0.4987 | 0.967 | 0.993 | 0.70 |  |
|  | lr=1e-05 | 0.7924 | 0.7980 | 0.4991 | 0.982 | 0.992 | 0.69 |  |
| NG | lr=0.001 | 0.7325 | 0.7322 | 0.4997 | 0.693 | 0.990 | 0.03 | 0.790 / 0.798 / 0.499 / 0.06 |
|  | lr=0.0001 | 0.7919 | 0.7967 | 0.5011 | 0.984 | 0.986 | 0.03 |  |
|  | lr=1e-05 | 0.7944 | 0.7979 | 0.4995 | 0.983 | 0.991 | 0.03 |  |
| ANG | lr=0.001 | 0.3559 | 0.3575 | 0.5165 | 0.073 | 0.941 | 1.26 | 0.792 / 0.800 / 0.497 / 2.21 |
|  | lr=0.0001 | 0.7839 | 0.7804 | 0.4950 | 0.935 | 0.996 | 1.25 |  |
|  | lr=1e-05 | 0.7937 | 0.7969 | 0.4960 | 0.988 | 0.999 | 1.25 |  |
| UNSIR | lr=0.001 | 0.7947 | 0.7956 | 0.5006 | 0.989 | 0.987 | 0.73 | 0.790 / 0.803 / 0.501 / 1.24 |
|  | lr=0.0001 | 0.7900 | 0.8024 | 0.5007 | 0.962 | 0.987 | 0.72 |  |
|  | lr=1e-05 | 0.7914 | 0.8016 | 0.5007 | 0.968 | 0.987 | 0.72 |  |
| CF-k | lr=0.001 | 0.7924 | 0.7977 | 0.4973 | 0.983 | 0.997 | 0.66 | 0.788 / 0.799 / 0.500 / 1.11 |
|  | lr=0.0001 | 0.7890 | 0.8033 | 0.4998 | 0.957 | 0.990 | 0.65 |  |
|  | lr=1e-05 | 0.7924 | 0.7989 | 0.5033 | 0.979 | 0.979 | 0.65 |  |
| EU-k | lr=0.001 | 0.7901 | 0.7995 | 0.5005 | 0.971 | 0.988 | 6.48 | 0.789 / 0.799 / 0.498 / 1.98 |
|  | lr=0.0001 | 0.7900 | 0.7994 | 0.5005 | 0.971 | 0.988 | 6.48 |  |
|  | lr=1e-05 | 0.7479 | 0.7539 | 0.4989 | 0.775 | 0.992 | 6.44 |  |
| SalUn | lr=0.001 | 0.7929 | 0.7976 | 0.4982 | 0.985 | 0.994 | 0.58 | 0.790 / 0.798 / 0.498 / 1.10 |
|  | lr=0.0001 | 0.7895 | 0.8001 | 0.4976 | 0.967 | 0.996 | 0.58 |  |
|  | lr=1e-05 | 0.7926 | 0.7980 | 0.4998 | 0.982 | 0.990 | 0.58 |  |
| BT | lr=0.001 | 0.7935 | 0.7982 | 0.4982 | 0.984 | 0.994 | 1.16 | 0.790 / 0.803 / 0.496 / 2.38 |
|  | lr=0.0001 | 0.7917 | 0.7968 | 0.4995 | 0.983 | 0.991 | 1.16 |  |
|  | lr=1e-05 | 0.7943 | 0.7972 | 0.4958 | 0.986 | 0.998 | 1.15 |  |
| SCRUB | lr=0.001 | 0.7934 | 0.8016 | 0.4982 | 0.974 | 0.995 | 0.64 | 0.793 / 0.800 / 0.499 / 1.10 |
|  | lr=0.0001 | 0.7936 | 0.8008 | 0.4980 | 0.977 | 0.995 | 0.64 |  |
|  | lr=1e-05 | 0.7903 | 0.8028 | 0.4986 | 0.962 | 0.993 | 0.64 |  |
| FF | alpha=1e-06 | 0.7787 | 0.7888 | 0.4966 | 0.944 | 0.999 | 0.13 | 0.776 / 0.778 / 0.499 / 0.08 |
|  | alpha=1e-07 | 0.7862 | 0.7962 | 0.5025 | 0.969 | 0.982 | 0.07 |  |
|  | alpha=1e-08 | 0.7659 | 0.7816 | 0.4988 | 0.889 | 0.993 | 0.07 |  |
| SSD | lr=0.001 | 0.7938 | 0.7976 | 0.4982 | 0.986 | 0.995 | 0.57 | 0.790 / 0.799 / 0.498 / 1.10 |
|  | lr=0.0001 | 0.7938 | 0.7976 | 0.4991 | 0.986 | 0.992 | 0.57 |  |
|  | lr=1e-05 | 0.7938 | 0.7976 | 0.4979 | 0.986 | 0.995 | 0.57 |  |

## What I noticed

On Adult, utility and efficacy barely separate the methods. Every well-behaved configuration lands at F1 ≈ .79, F1 forget ≈ .80 and UMIA ≈ .50, the same tight band as the paper, and M_U and M_E sit between .96 and 1.00 for nearly everything. So if you compute LUMA on this dataset, the ranking comes almost entirely from the efficiency term.

That makes the runtime differences important. My Gold model trained in 1.28 min against 2.08 min in the paper, which shifts every method's efficiency ratio. EU-k took 6.48 min against their 1.98. It's the only method configured for 10 epochs instead of 1, so it's the most exposed to hardware differences. The paper also gives EU-k its lowest LUMA (.408) because of efficiency, so the direction agrees, but the gap is much bigger here, and a LUMA computed on this machine won't match theirs row by row.

The original model records a runtime of about 0.0001 s, because the Identity "unlearner" does nothing. The paper gives the original model its training time instead (2.08 min, the same as Gold). If you compute LUMA yourself, match that convention, or the original model gets a huge efficiency advantage it hasn't earned.

Selective Synaptic Dampening does nothing on Adult. All three SSD configurations produce predictions identical to the original model, down to ten decimal places in F1. With `dampening_constant=0.1` the selection step apparently flags no weights. The paper's SSD row is also indistinguishable from the original, so this isn't a problem with my run, but any score SSD gets here is really a score for its runtime.

Over-unlearning shows up clearly. AdvancedNegGrad at lr=1e-3 collapses F1 from .794 to .356, and NegGrad at the same rate drops to .733. Both damage the model instead of removing the forget set's influence. A metric that just rewards a lower forget score would count these as successes. LUMA's terms use the absolute distance to Gold, so they penalise being worse than Gold and being "better" at forgetting than Gold in the same way, and these two runs get M_U of .073 and .693.

I left the efficiency term and LUMA itself out of the table on purpose. LUMA isn't implemented in the ERASURE repo (the configs dump its ingredients to JSON), so you have to compute it yourself. On top of that, in a single-process run ERASURE's `CudaPeak_MB` keeps growing through the run: EU-k runs 185x longer than NegGrad but reports mid-sequence memory. Peak memory then depends on where a method sits in the config rather than on what it costs. Resetting the peak counter between unlearners (`torch.cuda.reset_peak_memory_stats()`) or running each configuration in its own process should fix that.

## Repository layout

```
scripts/
  install.sh        install ERASURE's dependencies, keeping the cu130 torch build
  run_adult.sh      run the Adult 1-MLP, 5% forget, seed 0 config
  summarize.py      raw results -> table + CSV, with M_U / M_E against Gold
results/
  adult_1mlp_05pct_seed0.json              ERASURE's raw output (38 records)
  adult_1mlp_05pct_seed0_summary.csv       the table above
logs/
  install.log
  adult_1mlp_05pct_seed0.log               full run log, including the teardown segfault
env/
  pip-freeze.txt    the Python environment
```

The raw results file is in ERASURE's own format: JSON objects separated by commas, with a trailing comma, so it isn't valid JSON on its own. `summarize.py` handles that.

## Environment

- NVIDIA DGX Spark: GB10 Grace Blackwell GPU, aarch64, about 119 GiB unified CPU/GPU memory, Ubuntu 24.04
- Python 3.12.3, PyTorch 2.14.0+cu130
- `env/pip-freeze.txt` is a snapshot of the environment taken after the run. It includes a few packages I installed later for other work (for example `rouge`) that ERASURE doesn't use.

## References

[1] A. D'Angelo, C. Savelli, G. Tagliente, F. Giobergia, E. Baralis, G. Stilo. *How to Make Reproducible Research in Machine Unlearning with ERASURE.* IJCAI-25, Demo Track, 2025. https://doi.org/10.24963/ijcai.2025/1255

[2] A. D'Angelo, C. Savelli, G. Tagliente, F. Giobergia, E. Baralis, G. Stilo. *ERASURE: A Modular and Extensible Framework for Machine Unlearning.* CIKM '25, 2025. https://doi.org/10.1145/3746252.3761627

[3] D'Angelo et al. *On the Evaluation of Machine Unlearning Methods: A Multi-domain Classification Benchmark.* Machine Learning 115(161), 2026 (open access). https://doi.org/10.1007/s10994-026-07094-y

```bibtex
@inproceedings{ijcai2025p1255,
  title     = {How to Make Reproducible Research in Machine Unlearning with ERASURE},
  author    = {D'Angelo, Andrea and Savelli, Claudio and Tagliente, Gabriele and Giobergia, Flavio and Baralis, Elena and Stilo, Giovanni},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  pages     = {11025--11029},
  year      = {2025},
  note      = {Demo Track},
  doi       = {10.24963/ijcai.2025/1255}
}

@inproceedings{10.1145/3746252.3761627,
  title     = {ERASURE: A Modular and Extensible Framework for Machine Unlearning},
  author    = {D'Angelo, Andrea and Savelli, Claudio and Tagliente, Gabriele and Giobergia, Flavio and Baralis, Elena and Stilo, Giovanni},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '25},
  pages     = {6346--6350},
  year      = {2025},
  doi       = {10.1145/3746252.3761627}
}
```

## License

The scripts and notes in this repo are MIT licensed. ERASURE itself is released by its authors under CC0 1.0, and the results here were produced with it.
