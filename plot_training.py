#!/usr/bin/env python3
"""Plot e_loss, e_selection_hit_rate, e_dist_to_good from a training log."""

import sys
import matplotlib.pyplot as plt

MAX_SEGMENTS = 5
KEYS = ["e_loss", "e_hit", "e_dist", "t_loss", "t_hit", "t_dist"]
PREFIXES = {
    "e_loss ": "e_loss",
    "e_selection_hit_rate ": "e_hit",
    "e_dist_to_good ": "e_dist",
    "t_loss ": "t_loss",
    "t_selection_hit_rate ": "t_hit",
    "t_dist_to_good ": "t_dist",
}

def new_segment():
    return {k: [] for k in KEYS}

def parse_logs(paths):
    """Return dict of key -> list of (offset, values) segments."""
    segments = {k: [] for k in KEYS}
    cur = new_segment()
    offset = 0  # global x position

    def flush():
        nonlocal cur
        for k in KEYS:
            if cur[k]:
                segments[k].append((offset, cur[k]))
        n = max((len(cur[k]) for k in KEYS), default=0)
        return n

    for path in paths:
        # new file -> flush previous segment
        n = flush()
        offset += n
        cur = new_segment()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Loop took"):
                    n = flush()
                    offset += n
                    cur = new_segment()
                    continue
                for prefix, key in PREFIXES.items():
                    if line.startswith(prefix):
                        cur[key].append(float(line.split()[1]))
                        break

    flush()
    return segments

def plot_segments(ax, segments, style, label, **kwargs):
    """Plot list of (offset, values) segments. Return first handle for legend."""
    handle = None
    for offset, vals in segments[:MAX_SEGMENTS]:
        xs = list(range(offset, offset + len(vals)))
        h, = ax.plot(xs, vals, style, markersize=3, label=label if handle is None else None, **kwargs)
        if handle is None:
            handle = h
    return handle

def main():
    paths = sys.argv[1:]
    segments = parse_logs(paths)

    fig, ax1 = plt.subplots(figsize=(8, 3))

    handles = []

    ax2 = ax1.twinx()

    # h = plot_segments(ax2, segments["e_hit"], "o-", "e_selection_hit_rate"); handles.append(h)
    h = plot_segments(ax2, segments["e_dist"], "o-", "validation d2g", color="tab:blue"); handles.append(h)
    h = plot_segments(ax1, segments["e_loss"], "s-", "validation loss", color="tab:blue", alpha=0.5); handles.append(h)

    # highlight the minimum validation d2g in each segment
    best_handle = None
    for offset, vals in segments["e_dist"][:MAX_SEGMENTS]:
        if vals:
            best_i = min(range(len(vals)), key=lambda i: vals[i])
            h, = ax2.plot(offset + best_i, vals[best_i], "*", markersize=8, color="black", zorder=5,
                          label="best model (Mode I)" if best_handle is None else None)
            if best_handle is None:
                best_handle = h
    if best_handle is not None:
        handles.append(best_handle)

    h = plot_segments(ax2, segments["t_dist"], "o--", "train d2g", color="tab:red"); handles.append(h)
    h = plot_segments(ax1, segments["t_loss"], "s--", "train loss", color="tab:red", alpha=0.5); handles.append(h)

    ax1.set_xlabel("training round")
    ax1.set_ylabel("loss")
    ax1.set_ylim(top=4.2)
    # ax1.set_ylim(bottom=2.5)
    ax2.set_ylabel("distance to good (d2g)")
    ax2.set_ylim(top=0.08)
    # ax2.set_ylim(bottom=0.02)

    handles = [h for h in handles if h is not None]
    ax1.legend(handles=handles, loc="best", ncol=2)

    plt.savefig("training_metrics.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to training_metrics.pdf")

if __name__ == "__main__":
    main()
