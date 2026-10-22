# ForgeRT SqueezeNet 1.1 CPU Latency Baseline

**Run:** 2026-10-21, 20:57-20:59 (local). **Scope:** ForgeRT CPU path only, batch 1, one model, one
machine, one session. Raw data: [`SQUEEZENET_CPU_LATENCY.json`](SQUEEZENET_CPU_LATENCY.json).

> **Read this first.** This baseline was taken on a laptop **running on battery, under about 58%
> background CPU load**, and one diagnostic launch showed the CPU at only **41-79% of nominal
> performance**. The median inference time was **5.75 s**, about **4.5x slower** than the roughly
> 1.3 s measured in earlier sessions of this project, with identical code and binary. The machine
> state during those earlier runs was not recorded, so the cause of the difference is **not
> established**; the evidence below is consistent with throttling and contention, nothing more. Treat
> these numbers as a baseline for *this machine state*, not as ForgeRT's capability on an idle,
> plugged-in machine.

## Results

3 warm-up launches (discarded) then **20 measured launches**. All times in milliseconds.

| Metric | Median | Min | Max | Mean | Stdev | Q1 | Q3 |
|---|---|---|---|---|---|---|---|
| **End-to-end process latency** (wall clock) | **5912.76** | 4766.06 | 6833.24 | 5886.01 | 557.62 | 5563.06 | 6265.12 |
| **Inference only** (runner-reported) | **5747.64** | 4617.20 | 6765.15 | 5755.82 | 567.60 | 5464.81 | 6139.30 |
| Model import (runner-reported) | 87.63 | 35.97 | 115.37 | 77.71 | 23.92 | 60.70 | 93.93 |
| Other (process start/exit, file I/O, spawn) | 52.75 | 32.13 | 74.77 | 52.48 | 11.00 | 45.72 | 57.53 |

- **Spread is large:** inference ranged from 4,617 to 6,765 ms, i.e. (max-min)/median = 37.4%
  (coefficient of variation 9.9%). A comparison of two runs on this machine cannot resolve differences
  smaller than that spread.
- **No trend:** median inference was 5,748 ms in the first 10 measured launches and 5,702 ms in the
  last 10. The three warm-up launches (5,631 / 6,189 / 6,513 ms) fall in the same range, so warm-up
  showed no separable cold-start effect.
- **Same computation every time:** all 23 outputs are byte-identical (checked by SHA-256).

## What each number means

`forgert_run` performs **exactly one inference per process**, so repetition here is
**process-level**: the benchmark launches the runner 23 times. Nothing in the runtime was modified.

| Metric | What it includes | What it excludes |
|---|---|---|
| **End-to-end process latency** | Everything from the launcher's `subprocess.run()` call to its return: process creation, runtime/DLL load, `.fgx` import, reading the 602,112-byte input, one inference, writing the 4,000-byte output, process exit, output capture | The one-time `.onnx` to `.fgx` conversion, input generation |
| **Inference only** | The single `model->run()` call: graph ordering, allocating every intermediate tensor, executing all 66 nodes, freeing them | Process start, import, input read, output write |
| **Model import** | `ImportedModel::load()`: read the 4.96 MB `.fgx`, build the graph and weight tensors | Everything else |
| **Other** | Wall minus import minus inference: process start and exit, file reads and writes, launcher overhead | The above |

**Important limits of "inference only".** It is the *first and only* inference in its process, so it
is a **cold-in-process** figure: allocator growth, page faults on freshly loaded weights, and cold
caches are all inside it. It is **not** steady-state latency for a long-lived service. Measuring that
needs an in-process loop, which would mean changing the runtime, and I did not. The timed call also
includes per-call allocation of all intermediate tensors, because the current `Graph::execute` does
that on every call. The "warm-up" only warms the operating system's file cache and CPU state, not
anything inside the process.

## Command

From the repository root, with the project already built (`build\apps\forgert_run.exe` present):

```powershell
python tools\benchmark_squeezenet_cpu.py --runner build\apps\forgert_run.exe `
    --warmup 3 --runs 20 --json docs\SQUEEZENET_CPU_LATENCY.json
```

Each launch is `forgert_run <squeezenet1.1-7.fgx> <input.bin> <output.bin>`. The input is a float32
`[1,3,224,224]` tensor drawn from N(0,1) with `numpy.random.default_rng(20261020)`. Timing uses
`time.perf_counter_ns`; import and inference times are parsed from the runner's own stderr (printed
with 6 significant digits). The script exits non-zero if any output differs between launches.

## Machine and build

| Item | Value |
|---|---|
| CPU | Intel Core i5-9300H, 4 cores / 8 threads, 2.4 GHz base (reported max clock 2400 MHz) |
| Memory | 31.8 GiB, about 16.6 GiB free at the time |
| System | HP Pavilion Gaming Laptop 15-dk0xxx |
| OS | Windows 11 Home Single Language, build 26300 |
| Power plan | "HP Recommended" (not changed) |
| Python / NumPy | 3.14.3 / 2.4.3 |
| Compiler / flags | MSVC 14.39, Release: `/O2 /Ob2 /DNDEBUG /EHsc /GR`, C++17 |
| Parallelism | None: no threads, OpenMP, or parallel algorithms in the runtime sources (0 matches searched in `include/`, `src/`, `apps/`); the convolution is a plain direct loop |
| Binary | `forgert_run.exe`, 213,504 bytes, SHA-256 prefix `51997d9d1e753112`; build up to date (`ninja -n`: no work to do) |
| Model | `models/squeezenet1.1-7.onnx`, SHA-256 `1eeff551...4088` (verified), converted to a 4,962,820-byte `.fgx` |
| Git | HEAD `7fcd386`; 2 tracked `.kiro` documentation files modified, none of them code |

### Conditions during the run (not controlled)

| Moment | System CPU load | Battery | Instantaneous clock |
|---|---|---|---|
| Just before (20:57:00) | 58% | discharging, 71% | 2400 MHz |
| Just after (20:59:21) | 59% | discharging, 68% | 2400 MHz |

Running processes included Opera, Edge WebView2, and Kiro; I did not close or change anything.
The instantaneous clock readings are single samples and fluctuate (an earlier sample today read
1500 MHz), so they are weak evidence on their own.

**Diagnostic launch (one extra run, not in the statistics above).** To understand the slowness, I
launched the runner once more while sampling Windows' `% Processor Performance` counter once per
second. It read **79, 65, 51, 41** during the launch (53 and 40 while idle just before). That launch
took 4,581 ms of inference. This suggests the CPU was running well below nominal speed, but one
sample series on one launch cannot say why (battery policy, thermal limits, and contention from the
background load are all possible), and I did not test any of them.

## Comparison with earlier figures in this repo

| Source | Figure | Conditions recorded |
|---|---|---|
| `README.md`, `docs/ONNX_IMPORT.md` ("about 1.3 s per inference") | single runs | not recorded |
| `tools/validate_squeezenet.py` run, 2026-10-20 | 1,270-1,392 ms wall per input (10 inputs) | not recorded |
| This baseline | 5,913 ms median wall | battery, 58% background load, reduced CPU performance |

The earlier documents are **not updated** by this task. They remain true as descriptions of those
runs, but they should not be read as a typical or guaranteed latency. Whether to qualify them is your
decision.

## Limitations

- **Machine state dominates.** One battery-powered, loaded laptop session. No plugged-in or idle run
  was taken, so the size of the state effect is unknown.
- **Cold-in-process, not steady state** (see above). The runtime has no in-process repeat mode.
- **Single input and shape**, one model, batch 1, no CUDA.
- **No comparison with ONNX Runtime latency.** It was not measured, so no faster or slower claim
  is made.
- **Process-level numbers include some operating-system noise** that an in-process timer would not.
- **Launcher overhead** (Python `subprocess` creation and pipe capture) is inside the end-to-end figure.
- **Paths in the JSON are sanitized:** the runner path is repo-relative and the per-launch temp
  files appear as `<workdir>/...`; no absolute paths or user name are stored. All raw timing samples,
  hashes and statistics are unchanged.
- Process priority and CPU affinity were left at defaults.

## To get a cleaner baseline later (not done here)

Plug in AC power, close browsers and other heavy applications, confirm the CPU is not throttling, and
rerun the same command. Compare against this file only after noting both machine states.

## Files

- `tools/benchmark_squeezenet_cpu.py` (new): the benchmark.
- `docs/SQUEEZENET_CPU_LATENCY.json` (new): raw per-launch measurements, statistics, environment.
- `docs/SQUEEZENET_CPU_LATENCY.md` (new): this report.

No C++, CUDA, importer, operator, scheduler, or existing test file was changed.