#!/usr/bin/env python3
"""
Generate publication-ready visual comparisons for the dynamic Abilene runs.

The script intentionally depends only on numpy + Pillow so it can run on the
local machine or cluster without requiring matplotlib.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ["resilient_temporal", "vanilla_temporal", "snapshot_baseline"]
MODEL_LABELS = {
    "resilient_temporal": "Resilient temporal",
    "vanilla_temporal": "Vanilla temporal",
    "snapshot_baseline": "Snapshot baseline",
}
MODEL_COLORS = {
    "resilient_temporal": "#CC6677",
    "vanilla_temporal": "#4477AA",
    "snapshot_baseline": "#228833",
}
MODEL_MARKERS = {
    "resilient_temporal": "circle",
    "vanilla_temporal": "square",
    "snapshot_baseline": "triangle",
}
MODEL_LINE_STYLES = {
    "resilient_temporal": "solid",
    "vanilla_temporal": "dash",
    "snapshot_baseline": "dot",
}

METRIC_ORDER = ["current", "combined", "expected_failure", "worst_failure"]
METRIC_LABELS = {
    "current": "Current normalized MLU",
    "combined": "Combined resilience objective",
    "expected_failure": "Expected future-failure MLU",
    "worst_failure": "Worst single-link degradation",
}
METRIC_SHORT_LABELS = {
    "current": "Current",
    "combined": "Combined",
    "expected_failure": "Expected failure",
    "worst_failure": "Worst failure",
}

ALL_PERCENTILES = [25, 50, 75, 90, 95, 99, 100]
CHART_PERCENTILES = [50, 75, 90, 95, 99, 100]

BG = "#F7F8FA"
PANEL = "#FFFFFF"
INK = "#111827"
MUTED = "#5C6470"
SUBTLE = "#E6EAF0"
GRID = "#E9EDF3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize dynamic Abilene MLU and resilience results."
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        default=Path("results/dynamic_abilene/4sp/0/resilience_compare_all"),
        help="Directory produced by compare_all_dynamic_abilene_local_m3.sh.",
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
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        ]
    elif bold:
        candidates = [
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
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

    if style == "dash":
        pattern = (16, 8)
    else:
        pattern = (3, 8)

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
        seg_len = pattern[0] if draw_segment else pattern[1]
        end = min(length, pos + seg_len)
        if draw_segment:
            start_pt = (x1 + ux * pos, y1 + uy * pos)
            end_pt = (x1 + ux * end, y1 + uy * end)
            draw.line((start_pt[0], start_pt[1], end_pt[0], end_pt[1]), fill=fill, width=width)
        pos = end
        draw_segment = not draw_segment


def draw_marker(draw, xy, fill, marker="circle", size=7):
    x, y = xy
    if marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=fill)
    elif marker == "triangle":
        draw.polygon(
            [
                (x, y - size - 1),
                (x - size - 1, y + size),
                (x + size + 1, y + size),
            ],
            fill=fill,
        )
    else:
        draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)


def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def load_values(path):
    if not path.exists():
        raise FileNotFoundError(path)

    values = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        values.append(float(raw))

    if not values:
        raise ValueError(f"No numeric values in {path}")

    return np.asarray(values, dtype=np.float64)


def find_metric_values_file(directory, metric):
    matches = sorted(directory.glob(f"*_{metric}_values.txt"))
    if not matches:
        raise FileNotFoundError(f"Missing {metric} values in {directory}")
    return matches[0]


def load_distributions(compare_root):
    data = {}
    for model in MODEL_ORDER:
        model_dir = compare_root / model
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Missing model directory {model_dir}. Run "
                "./compare_all_dynamic_abilene_local_m3.sh first."
            )

        data[model] = {}
        for metric in METRIC_ORDER:
            path = find_metric_values_file(model_dir, metric)
            data[model][metric] = {
                "values": load_values(path),
                "path": path,
            }

    return data


def compute_stats(values):
    stats = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

    for pct in ALL_PERCENTILES:
        key = "max" if pct == 100 else f"p{pct}"
        stats[key] = float(np.percentile(values, pct))

    return stats


def compute_all_stats(data):
    stats = {}
    for model in MODEL_ORDER:
        stats[model] = {}
        for metric in METRIC_ORDER:
            stats[model][metric] = compute_stats(data[model][metric]["values"])
    return stats


def write_summary_files(stats, out_dir):
    csv_path = out_dir / "dynamic_abilene_model_percentiles.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "model",
                "count",
                "mean",
                "std",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            ]
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

    md_path = out_dir / "dynamic_abilene_model_percentiles.md"
    lines = [
        "# Dynamic Abilene Model Comparison",
        "",
        "All metrics are normalized. The current metric is taken from the",
        "three-way resilience evaluator so resilient temporal, vanilla temporal,",
        "and snapshot baseline are compared through the same code path.",
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
    series = []
    for pct in CHART_PERCENTILES:
        key = "max" if pct == 100 else f"p{pct}"
        series.append(stats[model][metric][key])
    return series


def nice_y_bounds(values):
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-9:
        span = max(abs(hi), 1.0) * 0.1

    lo -= span * 0.10
    hi += span * 0.14

    if hi <= 1.6:
        step = 0.05
    elif hi <= 3.0:
        step = 0.10
    else:
        step = 0.25

    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    if lo < 0:
        lo = 0.0

    return lo, hi


def format_tick(value):
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_paper_legend(draw, x, y, font):
    cursor = x
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        draw_styled_line(
            draw,
            (cursor, y + 10),
            (cursor + 46, y + 10),
            fill=color,
            width=3,
            style=MODEL_LINE_STYLES[model],
        )
        draw_marker(
            draw,
            (cursor + 23, y + 10),
            fill=color,
            marker=MODEL_MARKERS[model],
            size=6,
        )
        draw_text(draw, (cursor + 54, y), MODEL_LABELS[model], font, INK)
        label_w, _ = text_size(draw, MODEL_LABELS[model], font)
        cursor += 54 + label_w + 36


def draw_paper_line_chart(
    draw,
    box,
    stats,
    metric,
    panel_label=None,
    y_label=None,
    x_label=None,
    show_legend=False,
):
    x, y, w, h = box
    label_font = find_font(17)
    tick_font = find_font(15)
    title_font = find_font(18, bold=True)
    legend_font = find_font(16)

    plot_left = x + 70
    plot_top = y + 48
    plot_right = x + w - 28
    plot_bottom = y + h - 62

    if panel_label:
        draw_text(draw, (plot_left, y + 6), panel_label, title_font, INK)

    if show_legend:
        draw_paper_legend(draw, plot_left + 290, y + 8, legend_font)

    all_vals = []
    for model in MODEL_ORDER:
        all_vals.extend(percentile_series(stats, model, metric))
    lo, hi = nice_y_bounds(all_vals)

    # Sparse horizontal grid, like a paper figure.
    for i in range(5):
        t = lo + (hi - lo) * i / 4
        py = plot_bottom - (t - lo) / (hi - lo) * (plot_bottom - plot_top)
        draw.line((plot_left, py, plot_right, py), fill="#D9DEE7", width=1)
        label = format_tick(t)
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
        vals = percentile_series(stats, model, metric)
        color = MODEL_COLORS[model]
        line_style = MODEL_LINE_STYLES[model]
        marker = MODEL_MARKERS[model]
        points = [
            (
                x_positions[i],
                plot_bottom - (vals[i] - lo) / (hi - lo) * (plot_bottom - plot_top),
            )
            for i in range(len(vals))
        ]

        for a, b in zip(points, points[1:]):
            draw_styled_line(draw, a, b, fill=color, width=3, style=line_style)
        for px, py in points:
            draw_marker(draw, (px, py), fill=color, marker=marker, size=5)

    if y_label:
        draw_text(draw, (plot_left, plot_top - 28), y_label, label_font, INK)
    if x_label:
        tw, _ = text_size(draw, x_label, label_font)
        draw_text(draw, ((plot_left + plot_right) / 2 - tw / 2, plot_bottom + 40), x_label, label_font, INK)


def draw_table_figure(stats, out_path):
    row_h = 42
    width = 1720
    height = 660
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    x0, y0 = 48, 42
    table_w = width - 96

    header_font = find_font(17, bold=True)
    cell_font = find_font(16)
    mono = find_font(16, mono=True)

    columns = [
        ("metric", 0, 250),
        ("model", 270, 315),
        ("mean", 625, 120),
        ("p50", 755, 120),
        ("p90", 885, 120),
        ("p95", 1015, 120),
        ("p99", 1145, 120),
        ("max", 1275, 120),
    ]

    draw.line((x0, y0 + 30, x0 + table_w, y0 + 30), fill=INK, width=2)
    draw.line((x0, y0 + 36, x0 + table_w, y0 + 36), fill=INK, width=1)
    for title, offset, _ in columns:
        draw_text(draw, (x0 + offset, y0), title, header_font, INK)

    y = y0 + 46
    for metric in METRIC_ORDER:
        metric_start_y = y
        for model in MODEL_ORDER:
            if model == MODEL_ORDER[0]:
                draw_text(draw, (x0, y + 8), METRIC_SHORT_LABELS[metric], cell_font, INK)

            draw_styled_line(
                draw,
                (x0 + 270, y + 20),
                (x0 + 310, y + 20),
                fill=MODEL_COLORS[model],
                width=3,
                style=MODEL_LINE_STYLES[model],
            )
            draw_marker(
                draw,
                (x0 + 290, y + 20),
                fill=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                size=5,
            )
            draw_text(draw, (x0 + 322, y + 8), MODEL_LABELS[model], cell_font, INK)

            row = stats[model][metric]
            vals = [row["mean"], row["p50"], row["p90"], row["p95"], row["p99"], row["max"]]
            for val, (_, offset, _) in zip(vals, columns[2:]):
                draw_text(draw, (x0 + offset, y + 8), f"{val:.4f}", mono, INK)

            y += row_h

        draw.line((x0, y - 5, x0 + table_w, y - 5), fill="#AEB7C2", width=1)
        metric_end_y = y - 5
        draw.line((x0 + 250, metric_start_y - 5, x0 + 250, metric_end_y), fill="#C9D0DA", width=1)

    img.save(out_path, quality=95)


def draw_current_figure(stats, out_path):
    img = Image.new("RGB", (1480, 820), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw_paper_line_chart(
        draw,
        (50, 34, 1370, 710),
        stats,
        "current",
        panel_label=None,
        y_label="normalized MLU",
        x_label="percentile",
        show_legend=True,
    )
    img.save(out_path, quality=95)


def draw_resilience_panels(stats, out_path):
    img = Image.new("RGB", (1660, 1120), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw_paper_legend(draw, 540, 28, find_font(16))

    panels = [
        ((40, 70, 760, 500), "combined", "(a) combined"),
        ((850, 70, 760, 500), "current", "(b) current MLU"),
        ((40, 590, 760, 500), "expected_failure", "(c) expected failure"),
        ((850, 590, 760, 500), "worst_failure", "(d) worst failure"),
    ]

    for box, metric, label in panels:
        draw_paper_line_chart(
            draw,
            box,
            stats,
            metric,
            panel_label=label,
            y_label=None,
            x_label="percentile",
            show_legend=False,
        )

    img.save(out_path, quality=95)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_distributions(args.compare_root)
    stats = compute_all_stats(data)

    csv_path, md_path = write_summary_files(stats, args.out_dir)

    current_path = args.out_dir / "current_normalized_mlu_percentiles.png"
    resilience_path = args.out_dir / "resilience_metric_percentiles.png"
    table_path = args.out_dir / "dynamic_abilene_percentile_table.png"

    draw_current_figure(stats, current_path)
    draw_resilience_panels(stats, resilience_path)
    draw_table_figure(stats, table_path)

    print("Wrote:")
    for path in [current_path, resilience_path, table_path, csv_path, md_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
