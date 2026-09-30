"""Render exported activity evidence; plots do not imply physical capacity."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def window(case, name):
    return next(w for w in case["summary"]["windows"] if w["name"] == name)


def render(source, directory):
    data = json.loads(source.read_text())
    cases = {c["name"]: c for c in data["cases"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    for ax, mode, active, percent in zip(axes, ("hvx", "hmx"), ("4377", "4481"), ("4352", "4480")):
        windows = [window(cases[f"{mode}-d{d}"], "dsp_measurement") for d in (0, 25, 50, 100)]
        reference = windows[-1]["metrics"][active]["time_weighted_mean"]
        x = [w["covered_measured_active_region_percent"] for w in windows]
        y = [100 * w["metrics"][active]["time_weighted_mean"] / reference for w in windows]
        sdk = [w["metrics"][percent]["time_weighted_mean"] for w in windows]
        ax.plot([0, 100], [0, 100], color="#bdc7d3", ls="--", lw=1.5, label="Elapsed-time reference")
        ax.plot(x, sdk, "s-", color="#c45428", lw=2, markersize=6, label="SDK percent (time-weighted)")
        ax.plot(x, y, "o-", color="#136c83", lw=2, markersize=6, label="Rate / continuous control")
        ax.set_title(mode.upper() + " · one DSP worker", loc="left", fontweight="bold")
        ax.set_xlim(-3, 104); ax.set_ylim(-3, 108)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Measured active-loop time in covered intervals (%)")
    axes[0].set_ylabel("Reported or normalized value (%)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("SDK percent does not measure the fraction of elapsed time spent computing", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.035, "Snapdragon 888 / v68 · 8 s per case · QDSP clock 1420.8 MHz · Complete intervals only\nNormalization is an empirical activity reference, not peak compute capacity or lane occupancy.", fontsize=9, color="#495364")
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.07, right=0.99, wspace=0.12)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"LOAD-CALIBRATION-20260930.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.2))
    for ax, name in zip(axes, ("inference-m1", "inference-m32")):
        case = cases[name]
        prefill, generation = window(case, "prefill"), window(case, "generation")
        origin = prefill["window_start_ns"]
        selected = [p for p in case["packets"] if p["end_ns"] >= origin and p["start_ns"] <= generation["window_end_ns"]]
        for identifier, label, color in (("4377", "HVX active", "#168575"), ("4481", "HMX active", "#305bbb")):
            x, y = [], []
            for packet in selected:
                x.extend([(packet["start_ns"]-origin)/1e9, (packet["end_ns"]-origin)/1e9])
                y.extend([packet["values"][identifier]]*2)
            ax.plot(x, y, label=label, color=color, lw=1.2)
        ax.axvspan(0, (prefill["window_end_ns"]-origin)/1e9, color="#e7ebef", zorder=-1)
        ax.set_xlim(0, (generation["window_end_ns"]-origin)/1e9)
        ax.set_ylim(bottom=0); ax.set_ylabel("SDK activity rate (MCPS)")
        ax.set_title("One sequence" if name.endswith("m1") else "32 independent sequences", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2); ax.legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Seconds since prefill started (phone CNTVCT / DSP QTimer)")
    fig.suptitle("Inference activity aligned with phone-local phase markers", x=0.08, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.025, "Gray = prefill; remaining span = generation including host sampling. MCPS is not useful MAC/s.\nQwen2.5-0.5B F16 · previously validated optimized hybrid CPU/cDSP backend · HMX Clock metric returned zero.", fontsize=9, color="#495364")
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.99, hspace=0.4)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"INFERENCE-ACTIVITY-20260930.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]))
