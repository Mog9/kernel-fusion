import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(results):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#ffffff")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

    ax_table = fig.add_subplot(gs[0, :])
    ax_time  = fig.add_subplot(gs[1, 0])
    ax_mem   = fig.add_subplot(gs[1, 1])
    ax_bw    = fig.add_subplot(gs[1, 2])

    ax_table.set_facecolor("#ffffff")
    ax_table.axis("off")

    rows = [
        ["Metric",            "Unfused",                        "Fused"],
        ["Time (ms)",         f"{results['time_unfused']:.3f}", f"{results['time_fused']:.3f}"],
        ["Memory moved (MB)", f"{results['mem_unfused']:.2f}",  f"{results['mem_fused']:.2f}"],
        ["Bandwidth (GB/s)",  f"{results['bw_unfused']:.2f}",  f"{results['bw_fused']:.2f}"],
        ["Global passes",     "8",               "2"],
        ["Kernel launches",   "5",                              "1"],
        ["Speedup",           "1.00x (baseline)",              f"{results['speedup']:.2f}x"],
    ]

    cell_colors = []
    for i in range(len(rows)):
        if i == 0:
            cell_colors.append(["#dddddd", "#dddddd", "#dddddd"])
        else:
            cell_colors.append(["#f9f9f9", "#fff0f0", "#f0fff4"])

    table = ax_table.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_text_props(color="#111111", fontweight="bold")
        elif col_idx == 0:
            cell.set_text_props(color="#555555")
        elif col_idx == 1:
            cell.set_text_props(color="#cc0000")
        else:
            cell.set_text_props(color="#007700")

    ax_table.set_title("ReLU + LayerNorm — Unfused vs Fused", color="#111111",
                       fontsize=14, fontweight="bold", pad=12)

    unfused_color = "#e05555"
    fused_color   = "#4caf50"
    bar_style     = {"width": 0.35, "edgecolor": "#aaaaaa", "linewidth": 0.8}

    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#fafafa")
        ax.spines[:].set_color("#cccccc")
        ax.tick_params(colors="#333333")
        ax.yaxis.label.set_color("#333333")
        ax.set_title(title, color="#111111", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=9, color="#333333")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Unfused", "Fused"], color="#333333", fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    def add_value_labels(ax, bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f"{h:.3f}", ha="center", va="bottom", color="#111111", fontsize=9)

    bars = ax_time.bar([0, 1], [results["time_unfused"], results["time_fused"]],
                       color=[unfused_color, fused_color], **bar_style)
    style_ax(ax_time, "Execution Time", "ms")
    add_value_labels(ax_time, bars)

    bars = ax_mem.bar([0, 1], [results["mem_unfused"], results["mem_fused"]],
                      color=[unfused_color, fused_color], **bar_style)
    style_ax(ax_mem, "Memory Moved", "MB")
    add_value_labels(ax_mem, bars)

    bars = ax_bw.bar([0, 1], [results["bw_unfused"], results["bw_fused"]],
                     color=[unfused_color, fused_color], **bar_style)
    style_ax(ax_bw, "Effective Bandwidth", "GB/s")
    add_value_labels(ax_bw, bars)

    plt.savefig("fusion_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → fusion_results.png")
    plt.show()