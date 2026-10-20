import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Verified measurements: GTX 1050 sm_61, i5-9300H, CUDA 12.6 Release
# Source: build\benchmarks\benchmark_cuda.exe (100 GPU iters, 100 CPU iters)
# Only including rows where the benchmark completed without timeout.

operators = [
    "ReLU\n[1M]",
    "Add\n[1M]",
    "Softmax\n[512,1024]",
    "LayerNorm\n[512,768]",
    "MatMul\n[512x512]",
]

cpu_ms   = [0.974,  1.373,  10.097,  3.935,  884.670]
kern_ms  = [0.173,  0.242,   2.831,  0.152,    2.000]
e2e_ms   = [2.482,  3.912,   5.758,  1.857,    3.130]

x = np.arange(len(operators))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

bars_cpu  = ax.bar(x - width, cpu_ms,  width, label="CPU",          color="#58a6ff", zorder=3)
bars_kern = ax.bar(x,         kern_ms, width, label="GPU kernel only", color="#3fb950", zorder=3)
bars_e2e  = ax.bar(x + width, e2e_ms,  width, label="GPU end-to-end (H2D+kernel+D2H)", color="#f78166", zorder=3)

ax.set_yscale("log")
ax.set_ylabel("Latency (ms, log scale)", color="#c9d1d9", fontsize=11)
ax.set_title(
    "CPU vs GPU Latency — GTX 1050 (sm_61) / i5-9300H\n"
    "CUDA 12.6, MSVC 14.39 Release, 100 iterations",
    color="#c9d1d9", fontsize=11, pad=12
)
ax.set_xticks(x)
ax.set_xticklabels(operators, color="#c9d1d9", fontsize=9)
ax.tick_params(axis="y", colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.yaxis.grid(True, color="#21262d", linestyle="--", linewidth=0.7, zorder=0)

legend = ax.legend(
    framefacecolor="#161b22", edgecolor="#30363d",
    labelcolor="#c9d1d9", fontsize=9, loc="upper left"
)

# Annotate speedup (CPU / E2E) above E2E bars
for i, (c, e) in enumerate(zip(cpu_ms, e2e_ms)):
    ratio = c / e
    label = f"{ratio:.1f}x" if ratio >= 1 else f"0.{int(ratio*10)}x"
    ax.text(x[i] + width, e * 1.3, label, ha="center", va="bottom",
            color="#c9d1d9", fontsize=7.5, fontweight="bold")

ax.text(0.99, 0.02,
    "Speedup labels = CPU / E2E (includes PCIe transfers).\n"
    "GPU kernel-only excludes transfer cost.",
    transform=ax.transAxes, ha="right", va="bottom",
    color="#8b949e", fontsize=7, style="italic")

plt.tight_layout()
plt.savefig(r"D:\Projects\ForgeRT\docs\images\benchmark_latency.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved docs/images/benchmark_latency.png")
