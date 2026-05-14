#!/usr/bin/env python3
"""
Generate simple paper figures for the five-model dynamic Abilene comparison.

The script reads the distributions produced by
scripts/compare_all_five_dynamic_abilene.py and writes percentile plots for
current MLU and the resiliency metrics. It intentionally uses only numpy and
Pillow to match the existing visualization script.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = [
    "vanilla_temporal",
    "resilient_temporal",
    "snapshot_baseline",
    "dote_adapter",
    "teal_adapter",
]
MODEL_LABELS = {
    "vanilla_temporal": "Vanilla temporal",
    "resilient_temporal": "Resilient temporal",
    "snapshot_baseline": "Snapshot HARP",
    "dote_adapter": "DOTE adapter",
    "teal_adapter": "TEAL adapter",
}
MODEL_COLORS = {
    "vanilla_temporal": "#4477AA",
    "resilient_temporal": "#CC6677",
    "snapshot_baseline": "#228833",
    "dote_adapter": "#EE7733",
    "teal_adapter": "#666666",
}
MODEL_MARKERS = {
    "vanilla_temporal": "square",
    "resilient_temporal": "circle",
    "snapshot_baseline": "triangle",
    "dote_adapter": "diamond",
    "teal_adapter": "x",
}
MODEL_LINE_STYLES = {
    "vanilla_temporal": "solid",
    "resilient_temporal": "dash",
    "snapshot_baseline": "dot",
    "dote_adapter": "longdash",
    "teal_adapter": "solid",
}

METRIC_ORDER = ["current", "combined", "expected_failure", "worst_failure"]
METRIC_SHORT_LABELS = {
    "current": "Current",
    "combined": "Combined",
    "expected_failure": "Expected failure",
    "worst_failure": "Worst failure",
}

ALL_PERCENTILES = [25, 50, 75, 90, 95, 99, 100]
CHART_PERCENTILES = [50, 75, 90, 95, 99, 100]

INK = "#111827"
GRID = "#D9DEE7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize five-model dynamic Abilene MLU and resilience results."
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        default=Path("results/dynamic_abilene/4sp/0/resilience_compare_all_five"),
        help="Directory produced by scripts/compare_all_five_dynamic_abilene.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/dynamic_abilene"),
        help="Where to write PNG/CSV/Markdown artifacts.",
    )
    return parser.parse_args()


def find_font(size, bold=False, mono=False):
    if mono:
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text(draw, xy, text, font, fill=INK, anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_styled_line(draw, p1, p2, fill, width=3, style="solid"):
    if style == "solid":
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=fill, width=width)
        return

    patterns = {
        "dash": (16, 8),
        "longdash": (24, 8),
        "dot": (3, 8),
    }
    on_len, off_len = patterns.get(style, patterns["dot"])
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return

    ux = dx / length
    uy = dy / length
    pos = 0.0
    draw_segment = True
    while pos < length:
        seg_len = on_len if draw_segment else off_len
        end = min(length, pos + seg_len)
        if draw_segment:
            draw.line(
                (
                    x1 + ux * pos,
                    y1 + uy * pos,
                    x1 + ux * end,
                    y1 + uy * end,
                ),
                fill=fill,
                width=width,
            )
        pos = end
        draw_segment = not draw_segment


def draw_marker(draw, xy, fill, marker="circle", size=6):
    x, y = xy
    if marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=fill)
    elif marker == "triangle":
        draw.polygon([(x, y - size - 1), (x - size - 1, y + size), (x + size + 1, y + size)], fill=fill)
    elif marker == "diamond":
        draw.polygon([(x, y - size - 1), (x - size - 1, y), (x, y + size + 1), (x + size + 1, y)], fill=fill)
    elif marker == "x":
        draw.line((x - size, y - size, x + size, y + size), fill=fill, width=3)
        draw.line((x - size, y + size, x + size, y - size), fill=fill, width=3)
    else:
        draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)


def load_values(path):
    if not path.exists():
        raise FileNotFoundError(path)
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No numeric values in {path}")
    return np.asarray(values, dtype=np.float64)


def load_distributions(compare_root):
    data = {}
    for model in MODEL_ORDER:
        data[model] = {}
        for metric in METRIC_ORDER:
            path = compare_root / f"{model}_{metric}_values.txt"
            data[model][metric] = load_values(path)
    return data


def compute_stats(values):
    out = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for pct in ALL_PERCENTILES:
        key = "max" if pct == 100 else f"p{pct}"
        out[key] = float(np.percentile(values, pct))
    return out


def compute_all_stats(data):
    return {
        model: {metric: compute_stats(data[model][metric]) for metric in METRIC_ORDER}
        for model in MODEL_ORDER
    }


def write_summary_files(stats, out_dir):
    csv_path = out_dir / "all_five_dynamic_abilene_percentiles.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["metric", "model", "count", "mean", "std", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
        )
        for metric in METRIC_ORDER:
            for model in MODEL_ORDER:
                row = stats[model][metric]
                writer.writerow(
                    [
                        metric,
                        model,
                        row["count"],
                        f"{row['mean']:.9f}",
                        f"{row['std']:.9f}",
                        f"{row['p25']:.9f}",
                        f"{row['p50']:.9f}",
                        f"{row['p75']:.9f}",
                        f"{row['p90']:.9f}",
                        f"{row['p95']:.9f}",
                        f"{row['p99']:.9f}",
                        f"{row['max']:.9f}",
                    ]
                )

    md_path = out_dir / "all_five_dynamic_abilene_percentiles.md"
    lines = [
        "# Five-model Dynamic Abilene Comparison",
        "",
        "| Metric | Model | Mean | P50 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        for model in MODEL_ORDER:
            row = stats[model][metric]
            lines.append(
                "| "
                f"{METRIC_SHORT_LABELS[metric]} | "
                f"{MODEL_LABELS[model]} | "
                f"{row['mean']:.4f} | "
                f"{row['p50']:.4f} | "
                f"{row['p90']:.4f} | "
                f"{row['p95']:.4f} | "
                f"{row['p99']:.4f} | "
                f"{row['max']:.4f} |"
            )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def percentile_series(stats, model, metric):
    out = []
    for pct in CHART_PERCENTILES:
        key = "max" if pct == 100 else f"p{pct}"
        out.append(stats[model][metric][key])
    return out


def nice_y_bounds(values):
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-9:
        span = max(abs(hi), 1.0) * 0.1
    lo -= 0.08 * span
    hi += 0.12 * span

    if hi <= 1.6:
        step = 0.05
    elif hi <= 3.0:
        step = 0.25
    elif hi <= 8.0:
        step = 0.5
    else:
        step = 1.0

    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    return max(0.0, lo), hi


def format_tick(value):
    if value >= 10:
        return f"{value:.0f}"
    if value >= 3:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_legend(draw, x, y, font, max_width):
    row_y = y
    cursor = x
    for model in MODEL_ORDER:
        label = MODEL_LABELS[model]
        label_w, _ = text_size(draw, label, font)
        item_w = 54 + label_w + 32
        if cursor > x and cursor + item_w > x + max_width:
            cursor = x
            row_y += 30
        color = MODEL_COLORS[model]
        draw_styled_line(
            draw,
            (cursor, row_y + 10),
            (cursor + 42, row_y + 10),
            fill=color,
            width=3,
            style=MODEL_LINE_STYLES[model],
        )
        draw_marker(draw, (cursor + 21, row_y + 10), color, MODEL_MARKERS[model], size=5)
        draw_text(draw, (cursor + 52, row_y), label, font, INK)
        cursor += item_w


def draw_line_chart(
    draw,
    box,
    stats,
    metric,
    panel_label=None,
    y_label=None,
    x_label="percentile",
    show_legend=False,
):
    x, y, w, h = box
    label_font = find_font(17)
    tick_font = find_font(15)
    title_font = find_font(18, bold=True)
    legend_font = find_font(16)

    plot_left = x + 70
    plot_top = y + 52
    plot_right = x + w - 30
    plot_bottom = y + h - 64

    if panel_label:
        draw_text(draw, (plot_left, y + 8), panel_label, title_font, INK)
    if show_legend:
        draw_legend(draw, plot_left + 250, y + 8, legend_font, plot_right - plot_left - 260)

    all_vals = []
    for model in MODEL_ORDER:
        all_vals.extend(percentile_series(stats, model, metric))
    lo, hi = nice_y_bounds(all_vals)

    for i in range(5):
        tick = lo + (hi - lo) * i / 4
        py = plot_bottom - (tick - lo) / (hi - lo) * (plot_bottom - plot_top)
        draw.line((plot_left, py, plot_right, py), fill=GRID, width=1)
        label = format_tick(tick)
        tw, th = text_size(draw, label, tick_font)
        draw_text(draw, (plot_left - tw - 10, py - th / 2), label, tick_font, INK)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)

    x_positions = []
    for idx, pct in enumerate(CHART_PERCENTILES):
        px = plot_left + idx / (len(CHART_PERCENTILES) - 1) * (plot_right - plot_left)
        x_positions.append(px)
        label = "max" if pct == 100 else f"p{pct}"
        tw, th = text_size(draw, label, tick_font)
        draw_text(draw, (px - tw / 2, plot_bottom + 12), label, tick_font, INK)

    for model in MODEL_ORDER:
        values = percentile_series(stats, model, metric)
        points = [
            (x_positions[i], plot_bottom - (values[i] - lo) / (hi - lo) * (plot_bottom - plot_top))
            for i in range(len(values))
        ]
        color = MODEL_COLORS[model]
        for left, right in zip(points, points[1:]):
            draw_styled_line(draw, left, right, fill=color, width=3, style=MODEL_LINE_STYLES[model])
        for point in points:
            draw_marker(draw, point, color, MODEL_MARKERS[model], size=5)

    if y_label:
        draw_text(draw, (plot_left, plot_top - 28), y_label, label_font, INK)
    if x_label:
        tw, _ = text_size(draw, x_label, label_font)
        draw_text(draw, ((plot_left + plot_right) / 2 - tw / 2, plot_bottom + 40), x_label, label_font, INK)


def draw_current_figure(stats, out_path):
    img = Image.new("RGB", (1660, 840), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw_line_chart(
        draw,
        (48, 34, 1540, 720),
        stats,
        "current",
        y_label="normalized MLU",
        x_label="percentile",
        show_legend=True,
    )
    img.save(out_path, quality=95)


def draw_resilience_panels(stats, out_path):
    img = Image.new("RGB", (1720, 1160), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw_legend(draw, 430, 26, find_font(16), 980)

    panels = [
        ((42, 76, 790, 500), "combined", "(a) combined"),
        ((890, 76, 790, 500), "current", "(b) current MLU"),
        ((42, 602, 790, 500), "expected_failure", "(c) expected failure"),
        ((890, 602, 790, 500), "worst_failure", "(d) worst failure"),
    ]
    for box, metric, label in panels:
        draw_line_chart(draw, box, stats, metric, panel_label=label, y_label=None, x_label="percentile")
    img.save(out_path, quality=95)


def draw_table_figure(stats, out_path):
    row_h = 38
    width = 1660
    height = 850
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    x0, y0 = 46, 42
    header_font = find_font(17, bold=True)
    cell_font = find_font(16)
    mono = find_font(16, mono=True)

    columns = [
        ("metric", 0),
        ("model", 250),
        ("mean", 610),
        ("p50", 740),
        ("p90", 870),
        ("p95", 1000),
        ("p99", 1130),
        ("max", 1260),
    ]

    draw.line((x0, y0 + 30, width - 46, y0 + 30), fill=INK, width=2)
    for title, offset in columns:
        draw_text(draw, (x0 + offset, y0), title, header_font, INK)

    y = y0 + 46
    for metric in METRIC_ORDER:
        metric_start_y = y
        for model in MODEL_ORDER:
            if model == MODEL_ORDER[0]:
                draw_text(draw, (x0, y + 8), METRIC_SHORT_LABELS[metric], cell_font, INK)

            draw_styled_line(
                draw,
                (x0 + 250, y + 20),
                (x0 + 290, y + 20),
                fill=MODEL_COLORS[model],
                width=3,
                style=MODEL_LINE_STYLES[model],
            )
            draw_marker(draw, (x0 + 270, y + 20), MODEL_COLORS[model], MODEL_MARKERS[model], size=5)
            draw_text(draw, (x0 + 304, y + 8), MODEL_LABELS[model], cell_font, INK)

            row = stats[model][metric]
            values = [row["mean"], row["p50"], row["p90"], row["p95"], row["p99"], row["max"]]
            for value, (_, offset) in zip(values, columns[2:]):
                draw_text(draw, (x0 + offset, y + 8), f"{value:.4f}", mono, INK)
            y += row_h

        draw.line((x0, y - 5, width - 46, y - 5), fill="#AEB7C2", width=1)
        draw.line((x0 + 228, metric_start_y - 5, x0 + 228, y - 5), fill="#C9D0DA", width=1)

    img.save(out_path, quality=95)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_distributions(args.compare_root)
    stats = compute_all_stats(data)
    csv_path, md_path = write_summary_files(stats, args.out_dir)

    current_path = args.out_dir / "all_five_current_normalized_mlu_percentiles.png"
    resilience_path = args.out_dir / "all_five_resilience_metric_percentiles.png"
    table_path = args.out_dir / "all_five_dynamic_abilene_percentile_table.png"

    draw_current_figure(stats, current_path)
    draw_resilience_panels(stats, resilience_path)
    draw_table_figure(stats, table_path)

    print("Wrote:")
    for path in [current_path, resilience_path, table_path, csv_path, md_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
