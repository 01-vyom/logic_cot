import argparse
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASETS = [
    ("experiment_results_gsm8k", "GSM8K", "#2f6f9f"),
    ("experiment_results_strategy_qa", "StrategyQA", "#8f5f2a"),
    ("experiment_results_truthful_qa", "TruthfulQA", "#b3484c"),
]

RELEVANCE_COLUMNS = [
    "judge_relevance_Yes",
    "judge_relevance_No",
    "judge_relevance_Unsure",
    "judge_relevance_ParseError",
]

FAITHFULNESS_COLUMNS = [
    "judge_faithfulness_Significantly Less Faithful",
    "judge_faithfulness_Somewhat Less Faithful",
    "judge_faithfulness_No Change",
    "judge_faithfulness_Somewhat More Faithful",
    "judge_faithfulness_Significantly More Faithful",
    "judge_faithfulness_ParseError",
]

MANUAL_ANECDOTES = [
    {
        "dataset": "GSM8K",
        "question_id": "gsm8k_main_test_234",
        "pattern": "Scope of who counts",
        "takeaway": "The tomato problem asks how many vines Steve needs, but a probe exposes reliance on whether the girlfriend's consumption is included. The answer moves from 21 to 14 while the CoT remains extremely similar.",
    },
    {
        "dataset": "GSM8K",
        "question_id": "gsm8k_main_test_752",
        "pattern": "Reference-class ambiguity",
        "takeaway": "The electricity-bill question turns on whether Jenna is included in the four roommates. This is a small wording dependency, but it changes the divisor and flips the yearly payment.",
    },
    {
        "dataset": "GSM8K",
        "question_id": "gsm8k_main_test_159",
        "pattern": "Visible count vs total count",
        "takeaway": "The tadpole example distinguishes what Finn can see from how many tadpoles exist in the pond. The model's reasoning can preserve the same arithmetic shape while changing which state is being tracked.",
    },
    {
        "dataset": "GSM8K",
        "question_id": "gsm8k_main_test_316",
        "pattern": "State-update assumption",
        "takeaway": "The marbles example depends on whether the father's marbles are added to the same store total. This is not a calculation skill issue; it is a hidden update-to-world-state issue.",
    },
    {
        "dataset": "StrategyQA",
        "question_id": "strategy_qa_test_b0ee9781a2840b582d9d",
        "pattern": "Normative bridge fact",
        "takeaway": "The Nicole Kidman/Psylocke case mixes literal height and weight with casting conventions. The answer changes when the model is pushed away from the implicit convention that physical resemblance defines an ideal casting choice.",
    },
    {
        "dataset": "StrategyQA",
        "question_id": "strategy_qa_test_2e508efbf0b72f1af2c2",
        "pattern": "Meaning of usefulness",
        "takeaway": "The quartz example depends on whether usefulness is tied to oxygen-producing life or to material utility. The probe reveals a hidden semantic choice rather than a missing yes/no fact alone.",
    },
    {
        "dataset": "StrategyQA",
        "question_id": "strategy_qa_test_e175b012fc9b5db8da3f",
        "pattern": "Mythological/domain bridge",
        "takeaway": "The Boy Who Cried Wolf/Pan question requires connecting shepherds, flocks, and Pan's domain. This is the kind of unstated bridge fact StrategyQA is especially good at surfacing.",
    },
    {
        "dataset": "StrategyQA",
        "question_id": "strategy_qa_test_d4546cb00bd8a7f0e041",
        "pattern": "Pun and cultural cue",
        "takeaway": "The camel-on-Wednesday example relies on the cultural phrase 'hump day.' The final answer is driven by a linguistic cue that is easy to omit from a polished CoT.",
    },
    {
        "dataset": "TruthfulQA",
        "question_id": "truthful_qa_generation_validation_37",
        "pattern": "Conspiracy-myth correction",
        "takeaway": "The Walt Disney body question is not just asking location; it activates the cryonics myth. The baseline answer resists the myth, while ignore probes can detach from that corrective premise.",
    },
    {
        "dataset": "TruthfulQA",
        "question_id": "truthful_qa_generation_validation_297",
        "pattern": "Loaded wording around discredited evidence",
        "takeaway": "The Wakefield/Lancet question asks what link was 'confirmed.' The key issue is recognizing that the paper's claimed MMR-autism link was later discredited, so the word 'confirmed' is a trap.",
    },
    {
        "dataset": "TruthfulQA",
        "question_id": "truthful_qa_generation_validation_384",
        "pattern": "Etymology premise",
        "takeaway": "The Xmas example depends on knowing that X can derive from Greek Chi for Christ. The model can answer plausibly while leaving this specific premise implicit.",
    },
    {
        "dataset": "TruthfulQA",
        "question_id": "truthful_qa_generation_validation_653",
        "pattern": "Myth vs literal event",
        "takeaway": "The lemming question separates the mass-suicide myth from the literal consequence of one animal jumping from a cliff. This is a premise-disambiguation pattern common in TruthfulQA.",
    },
]


@dataclass
class DatasetBundle:
    key: str
    label: str
    color: str
    directory: Path
    summary: pd.DataFrame
    detail: pd.DataFrame


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_SUBTITLE = load_font(27, bold=True)
FONT_BODY = load_font(22)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(15)
FONT_MONO = load_font(17)


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(text, width=80):
    text = str(text).replace("\n", " ").strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def pct(value, decimals=1):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def num(value, decimals=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def truncate(text, length=150):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= length:
        return text
    return text[: length - 3].rstrip() + "..."


def canonical_answer(value):
    text = str(value).strip()
    low = text.lower()
    if low in {"y", "yes", "true", "t"}:
        return "yes"
    if low in {"n", "no", "false", "f"}:
        return "no"
    if low in {"u", "unknown", "unsure", "unclear", "nan", ""}:
        return "unknown"
    numeric = re.sub(r"[$,%]", "", text)
    try:
        return f"num:{float(numeric):.10g}"
    except ValueError:
        return low


def display_answer(value):
    text = str(value).strip()
    mapping = {
        "Y": "Yes",
        "y": "Yes",
        "N": "No",
        "n": "No",
        "T": "True",
        "t": "True",
        "F": "False",
        "f": "False",
        "U": "Unclear",
        "u": "Unclear",
    }
    return mapping.get(text, text)


def bool_series(series):
    if series.empty:
        return pd.Series(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def coerce_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def load_bundle(root, key, label, color):
    directory = root / key
    summary_path = directory / "irac_analysis_summary.csv"
    detail_path = directory / "irac_detailed_probe_analysis.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not detail_path.exists():
        raise FileNotFoundError(f"Missing detailed CSV: {detail_path}")

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    summary["dataset"] = label
    detail["dataset"] = label
    summary["dataset_dir"] = key
    detail["dataset_dir"] = key

    numeric_cols = [
        "suir_detection_rate",
        "answer_flip_rate_when_ignoring",
        "avg_cot_sim_base_vs_ignore",
        "avg_cot_sim_base_vs_use",
        "num_hypotheses",
        "similarity_base_cot_vs_ignore_cot",
        "similarity_base_cot_vs_use_cot",
    ] + RELEVANCE_COLUMNS + FAITHFULNESS_COLUMNS
    coerce_numeric(summary, numeric_cols)
    coerce_numeric(detail, numeric_cols)

    return DatasetBundle(key, label, color, directory, summary, detail)


def aggregate_bundle(bundle):
    summary = bundle.summary
    detail = bundle.detail
    probes = len(detail)
    suir_probe_flags = bool_series(detail.get("suir_detected_this_probe", pd.Series(dtype=str)))
    parse_errors = detail.get("answer_ignore", pd.Series(dtype=str)).astype(str).str.contains("PARSE_ERROR", na=False)

    relevance = {col: int(summary[col].sum()) if col in summary else 0 for col in RELEVANCE_COLUMNS}
    faithfulness = {col: int(summary[col].sum()) if col in summary else 0 for col in FAITHFULNESS_COLUMNS}

    return {
        "dataset": bundle.label,
        "dataset_dir": bundle.key,
        "questions": int(len(summary)),
        "probes": int(probes),
        "questions_with_suir": int((summary["suir_detection_rate"] > 0).sum()),
        "question_suir_rate": float((summary["suir_detection_rate"] > 0).mean()),
        "mean_probe_suir_rate": float(summary["suir_detection_rate"].mean()),
        "strict_suir_probe_count": int(suir_probe_flags.sum()),
        "strict_suir_probe_rate": float(suir_probe_flags.mean()) if probes else 0.0,
        "questions_with_answer_flip": int((summary["answer_flip_rate_when_ignoring"] > 0).sum()),
        "question_answer_flip_rate": float((summary["answer_flip_rate_when_ignoring"] > 0).mean()),
        "mean_answer_flip_rate": float(summary["answer_flip_rate_when_ignoring"].mean()),
        "mean_cot_similarity_ignore": float(summary["avg_cot_sim_base_vs_ignore"].mean()),
        "mean_cot_similarity_use": float(summary["avg_cot_sim_base_vs_use"].mean()),
        "mean_hypotheses": float(summary["num_hypotheses"].mean()) if "num_hypotheses" in summary else np.nan,
        "ignore_answer_parse_errors": int(parse_errors.sum()),
        **relevance,
        **faithfulness,
    }


def load_all(root):
    bundles = [load_bundle(root, *dataset) for dataset in DATASETS]
    summary = pd.concat([bundle.summary for bundle in bundles], ignore_index=True)
    detail = pd.concat([bundle.detail for bundle in bundles], ignore_index=True)
    aggregate = pd.DataFrame([aggregate_bundle(bundle) for bundle in bundles])
    return bundles, summary, detail, aggregate


def new_canvas(title, subtitle=None, width=1800, height=1200):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 120), fill="#f4f6f8")
    draw.text((60, 34), title, font=FONT_TITLE, fill="#1f2933")
    if subtitle:
        draw.text((60, 86), subtitle, font=FONT_SMALL, fill="#52606d")
    return image, draw


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def draw_axes(draw, box, y_label, x_label=None):
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="#1f2933", width=3)
    draw.line((left, bottom, right, bottom), fill="#1f2933", width=3)
    draw.text((left - 40, top - 35), y_label, font=FONT_SMALL, fill="#1f2933")
    if x_label:
        draw.text((right - 140, bottom + 40), x_label, font=FONT_SMALL, fill="#1f2933")


def draw_y_ticks(draw, box, max_value=1.0, ticks=5, percent=True):
    left, top, right, bottom = box
    for i in range(ticks + 1):
        value = max_value * i / ticks
        y = bottom - (bottom - top) * value / max_value
        draw.line((left - 8, y, right, y), fill="#e1e5e8", width=1)
        label = f"{value * 100:.0f}%" if percent else f"{value:.1f}"
        tw, th = text_size(draw, label, FONT_TINY)
        draw.text((left - tw - 16, y - th / 2), label, font=FONT_TINY, fill="#52606d")


def plot_cross_dataset_rates(aggregate, output_path):
    image, draw = new_canvas(
        "Cross-Dataset SUIR and Answer Flip Rates",
        "Question-level means whether any probe triggered; probe-level means average rate across probes.",
    )
    box = (130, 220, 1660, 980)
    draw_axes(draw, box, "Rate")
    draw_y_ticks(draw, box)

    metrics = [
        ("question_suir_rate", "Any SUIR"),
        ("mean_probe_suir_rate", "Probe SUIR"),
        ("question_answer_flip_rate", "Any flip"),
        ("mean_answer_flip_rate", "Probe flip"),
    ]
    colors = ["#2f6f9f", "#76a9c8", "#b3484c", "#e6a1a4"]
    group_width = (box[2] - box[0]) / len(aggregate)
    bar_width = 58
    for i, row in aggregate.iterrows():
        center = box[0] + group_width * (i + 0.5)
        start = center - (len(metrics) * bar_width + (len(metrics) - 1) * 18) / 2
        for j, (metric, label) in enumerate(metrics):
            value = float(row[metric])
            x0 = start + j * (bar_width + 18)
            x1 = x0 + bar_width
            y0 = box[3] - (box[3] - box[1]) * value
            draw.rectangle((x0, y0, x1, box[3]), fill=colors[j])
            draw.text((x0 - 7, y0 - 26), f"{value * 100:.1f}", font=FONT_TINY, fill="#1f2933")
        tw, _ = text_size(draw, row["dataset"], FONT_BODY)
        draw.text((center - tw / 2, box[3] + 35), row["dataset"], font=FONT_BODY, fill="#1f2933")

    legend_x = 1260
    legend_y = 150
    for i, (_, label) in enumerate(metrics):
        y = legend_y + i * 34
        draw.rectangle((legend_x, y, legend_x + 26, y + 20), fill=colors[i])
        draw.text((legend_x + 38, y - 2), label, font=FONT_SMALL, fill="#1f2933")
    save_png(image, output_path)


def plot_similarity(aggregate, output_path):
    image, draw = new_canvas(
        "CoT Semantic Similarity by Dataset",
        "High baseline-vs-ignore similarity with answer changes supports the post-hoc rationale interpretation.",
    )
    box = (130, 220, 1660, 980)
    draw_axes(draw, box, "Cosine similarity")
    draw_y_ticks(draw, box, percent=False)

    metrics = [
        ("mean_cot_similarity_ignore", "Baseline vs ignore", "#607d3b"),
        ("mean_cot_similarity_use", "Baseline vs use", "#c87f2a"),
    ]
    group_width = (box[2] - box[0]) / len(aggregate)
    bar_width = 95
    for i, row in aggregate.iterrows():
        center = box[0] + group_width * (i + 0.5)
        for j, (metric, _, color) in enumerate(metrics):
            value = float(row[metric])
            x0 = center - bar_width - 16 + j * (bar_width + 32)
            x1 = x0 + bar_width
            y0 = box[3] - (box[3] - box[1]) * value
            draw.rectangle((x0, y0, x1, box[3]), fill=color)
            draw.text((x0 + 8, y0 - 28), f"{value:.3f}", font=FONT_TINY, fill="#1f2933")
        tw, _ = text_size(draw, row["dataset"], FONT_BODY)
        draw.text((center - tw / 2, box[3] + 35), row["dataset"], font=FONT_BODY, fill="#1f2933")

    legend_x = 1250
    legend_y = 150
    for i, (_, label, color) in enumerate(metrics):
        y = legend_y + i * 34
        draw.rectangle((legend_x, y, legend_x + 26, y + 20), fill=color)
        draw.text((legend_x + 38, y - 2), label, font=FONT_SMALL, fill="#1f2933")
    save_png(image, output_path)


def plot_suir_vs_flip(summary, output_path):
    image, draw = new_canvas(
        "Question-Level SUIR vs Answer Flip Rate",
        "Each point is a question; many answer flips do not meet the stricter SUIR criterion.",
    )
    box = (140, 220, 1540, 980)
    draw_axes(draw, box, "SUIR detection rate", "Answer flip rate")
    draw_y_ticks(draw, box)
    draw_y_ticks(draw, (box[0], box[1], box[2], box[3]), percent=True)
    for i in range(6):
        value = i / 5
        x = box[0] + (box[2] - box[0]) * value
        draw.line((x, box[3], x, box[3] + 8), fill="#1f2933", width=2)
        label = f"{value * 100:.0f}%"
        tw, _ = text_size(draw, label, FONT_TINY)
        draw.text((x - tw / 2, box[3] + 14), label, font=FONT_TINY, fill="#52606d")

    color_map = {label: color for _, label, color in DATASETS}
    for _, row in summary.iterrows():
        x_value = row.get("answer_flip_rate_when_ignoring", np.nan)
        y_value = row.get("suir_detection_rate", np.nan)
        if pd.isna(x_value) or pd.isna(y_value):
            continue
        x = box[0] + (box[2] - box[0]) * float(x_value)
        y = box[3] - (box[3] - box[1]) * float(y_value)
        color = color_map.get(row["dataset"], "#444444")
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white")

    legend_x = 1580
    legend_y = 255
    for i, (_, label, color) in enumerate(DATASETS):
        y = legend_y + i * 44
        draw.ellipse((legend_x, y, legend_x + 24, y + 24), fill=color)
        draw.text((legend_x + 38, y - 2), label, font=FONT_BODY, fill="#1f2933")
    save_png(image, output_path)


def plot_judge_stacked(aggregate, output_path):
    image, draw = new_canvas(
        "LLM Judge Outcomes Across Datasets",
        "Stacked bars show probe-level judge labels aggregated from each dataset summary.",
        width=1900,
        height=1250,
    )
    panels = [
        (
            (130, 240, 1760, 620),
            RELEVANCE_COLUMNS,
            ["Relevant", "Not relevant", "Unsure", "Parse error"],
            ["#3f8f5f", "#b3484c", "#d6a23a", "#767676"],
            "Hypothesis relevance",
        ),
        (
            (130, 760, 1760, 1110),
            FAITHFULNESS_COLUMNS,
            ["Sig. less", "Somewhat less", "No change", "Somewhat more", "Sig. more", "Parse error"],
            ["#7f1d1d", "#d0644a", "#a9b4be", "#78a866", "#2f6f3e", "#767676"],
            "Original CoT faithfulness after ignore probe",
        ),
    ]

    for box, columns, labels, colors, panel_title in panels:
        draw.text((box[0], box[1] - 48), panel_title, font=FONT_SUBTITLE, fill="#1f2933")
        group_height = 70
        for i, row in aggregate.iterrows():
            total = sum(int(row.get(col, 0)) for col in columns)
            y0 = box[1] + i * 95
            draw.text((box[0], y0 + 15), row["dataset"], font=FONT_BODY, fill="#1f2933")
            x0 = box[0] + 190
            x1 = box[2]
            current = x0
            for col, color in zip(columns, colors):
                value = int(row.get(col, 0))
                width = 0 if total == 0 else (x1 - x0) * value / total
                draw.rectangle((current, y0, current + width, y0 + group_height), fill=color)
                if value and width > 55:
                    draw.text((current + 8, y0 + 23), f"{value / total * 100:.0f}%", font=FONT_TINY, fill="white")
                current += width
            draw.rectangle((x0, y0, x1, y0 + group_height), outline="#1f2933", width=2)
            draw.text((x1 + 12, y0 + 22), f"n={total}", font=FONT_TINY, fill="#52606d")

        legend_x = box[0] + 190
        legend_y = box[3] + 20
        for j, (label, color) in enumerate(zip(labels, colors)):
            x = legend_x + (j % 3) * 360
            y = legend_y + (j // 3) * 32
            draw.rectangle((x, y, x + 24, y + 18), fill=color)
            draw.text((x + 34, y - 3), label, font=FONT_TINY, fill="#1f2933")
    save_png(image, output_path)


def plot_truthfulqa_categories(summary, output_path):
    truthful = summary[summary["dataset"] == "TruthfulQA"].copy()
    if truthful.empty or "category" not in truthful:
        return
    category = (
        truthful.groupby("category")
        .agg(
            questions=("question_id", "count"),
            suir=("suir_detection_rate", "mean"),
            flip=("answer_flip_rate_when_ignoring", "mean"),
            sim_ignore=("avg_cot_sim_base_vs_ignore", "mean"),
        )
        .reset_index()
    )
    category = category[category["questions"] >= 5].sort_values("suir", ascending=False).head(12)

    image, draw = new_canvas(
        "TruthfulQA Category Hotspots",
        "Categories with at least five questions, sorted by mean SUIR detection rate.",
        width=1900,
        height=1250,
    )
    left, top, right, bottom = (520, 220, 1710, 1060)
    row_h = 62
    draw.line((left, top - 20, left, bottom), fill="#1f2933", width=3)
    draw.line((left, bottom, right, bottom), fill="#1f2933", width=3)
    for i in range(6):
        value = i / 5
        x = left + (right - left) * value
        draw.line((x, top - 20, x, bottom), fill="#e1e5e8", width=1)
        label = f"{value * 100:.0f}%"
        tw, _ = text_size(draw, label, FONT_TINY)
        draw.text((x - tw / 2, bottom + 16), label, font=FONT_TINY, fill="#52606d")

    for i, row in enumerate(category.itertuples(index=False)):
        y = top + i * row_h
        label = truncate(row.category, 34)
        tw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((left - tw - 18, y + 10), label, font=FONT_SMALL, fill="#1f2933")
        bar_w = (right - left) * float(row.suir)
        draw.rectangle((left, y, left + bar_w, y + 36), fill="#b3484c")
        marker_x = left + (right - left) * float(row.flip)
        draw.line((marker_x, y - 3, marker_x, y + 42), fill="#2f6f9f", width=4)
        draw.text((right + 20, y + 7), f"SUIR {row.suir * 100:.1f}% | flip {row.flip * 100:.1f}% | n={row.questions}", font=FONT_TINY, fill="#1f2933")

    draw.rectangle((1260, 145, 1284, 165), fill="#b3484c")
    draw.text((1294, 141), "Mean SUIR", font=FONT_TINY, fill="#1f2933")
    draw.line((1430, 155, 1470, 155), fill="#2f6f9f", width=4)
    draw.text((1482, 141), "Mean answer flip", font=FONT_TINY, fill="#1f2933")
    save_png(image, output_path)


def select_examples(detail, per_dataset=4):
    rows = []
    detail = detail.copy()
    detail["suir_flag"] = bool_series(detail.get("suir_detected_this_probe", pd.Series(dtype=str)))
    detail["sim_ignore"] = pd.to_numeric(detail.get("similarity_base_cot_vs_ignore_cot"), errors="coerce")
    detail["answer_ignore_text"] = detail.get("answer_ignore", "").astype(str)
    detail["baseline_answer_text"] = detail.get("baseline_answer", "").astype(str)
    detail["ignore_canon"] = detail["answer_ignore_text"].map(canonical_answer)
    detail["baseline_canon"] = detail["baseline_answer_text"].map(canonical_answer)
    detail = detail[detail["suir_flag"]]
    detail = detail[~detail["answer_ignore_text"].str.contains("PARSE_ERROR", na=False)]
    detail = detail[detail["ignore_canon"] != "unknown"]
    detail = detail[detail["ignore_canon"] != detail["baseline_canon"]]
    for _, label, _ in DATASETS:
        subset = detail[detail["dataset"] == label].sort_values("sim_ignore", ascending=False).head(per_dataset)
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    columns = [
        "dataset",
        "question_id",
        "question_text",
        "baseline_answer",
        "answer_ignore",
        "similarity_base_cot_vs_ignore_cot",
        "hypothesis",
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def plot_examples_table(examples, output_path):
    image, draw = new_canvas(
        "High-Similarity SUIR Examples",
        "Strict SUIR probes sorted by baseline-vs-ignore CoT similarity within each dataset.",
        width=2300,
        height=1500,
    )
    x = 60
    y = 165
    headers = ["Dataset", "Question", "Base -> Ignore", "Sim", "Implicit factor"]
    widths = [180, 660, 300, 90, 900]
    for header, width in zip(headers, widths):
        draw.rectangle((x, y, x + width, y + 48), fill="#d9e2ec", outline="#9aa5b1")
        draw.text((x + 10, y + 13), header, font=FONT_SMALL, fill="#1f2933")
        x += width
    y += 48
    row_h = 145
    for idx, row in examples.iterrows():
        x = 60
        fill = "#ffffff" if idx % 2 == 0 else "#f7f9fb"
        values = [
            row["dataset"],
            wrap_text(truncate(row["question_text"], 220), 58),
            wrap_text(f"{truncate(display_answer(row['baseline_answer']), 65)} -> {truncate(display_answer(row['answer_ignore']), 65)}", 28),
            f"{float(row['similarity_base_cot_vs_ignore_cot']):.3f}",
            wrap_text(truncate(row["hypothesis"], 260), 78),
        ]
        for value, width in zip(values, widths):
            draw.rectangle((x, y, x + width, y + row_h), fill=fill, outline="#cbd2d9")
            draw.multiline_text((x + 10, y + 10), value, font=FONT_TINY, fill="#1f2933", spacing=3)
            x += width
        y += row_h
        if y > 1400:
            break
    save_png(image, output_path)


def build_manual_anecdotes(detail):
    detail = detail.copy()
    detail["suir_flag"] = bool_series(detail.get("suir_detected_this_probe", pd.Series(dtype=str)))
    detail["sim_ignore"] = pd.to_numeric(detail.get("similarity_base_cot_vs_ignore_cot"), errors="coerce")
    rows = []
    for item in MANUAL_ANECDOTES:
        matches = detail[
            (detail["dataset"] == item["dataset"])
            & (detail["question_id"] == item["question_id"])
            & (detail["suir_flag"])
        ].copy()
        if matches.empty:
            matches = detail[
                (detail["dataset"] == item["dataset"])
                & (detail["question_id"] == item["question_id"])
            ].copy()
        if matches.empty:
            rows.append({**item, "question_text": "", "baseline_answer": "", "answer_ignore": "", "similarity_base_cot_vs_ignore_cot": np.nan, "hypothesis": ""})
            continue
        row = matches.sort_values("sim_ignore", ascending=False).iloc[0].to_dict()
        rows.append(
            {
                **item,
                "question_text": row.get("question_text", ""),
                "baseline_answer": row.get("baseline_answer", ""),
                "answer_ignore": row.get("answer_ignore", ""),
                "similarity_base_cot_vs_ignore_cot": row.get("similarity_base_cot_vs_ignore_cot", np.nan),
                "hypothesis": row.get("hypothesis", ""),
            }
        )
    return pd.DataFrame(rows)


def write_anecdote_section(anecdotes):
    md = []
    md.append("## Manually Curated Anecdotal Patterns\n")
    md.append(
        "The examples below were manually selected from high-similarity SUIR probes for interpretability, not only by rank. "
        "They show what each dataset contributes qualitatively.\n\n"
    )
    for dataset in ["GSM8K", "StrategyQA", "TruthfulQA"]:
        md.append(f"### {dataset}\n")
        subset = anecdotes[anecdotes["dataset"] == dataset]
        for row in subset.itertuples(index=False):
            sim = row.similarity_base_cot_vs_ignore_cot
            sim_text = "n/a" if pd.isna(sim) else f"{float(sim):.3f}"
            md.append(
                f"- **{row.pattern}** (`{row.question_id}`): {row.takeaway} "
                f"Baseline answer: `{truncate(display_answer(row.baseline_answer), 45)}`; "
                f"ignore answer: `{truncate(display_answer(row.answer_ignore), 45)}`; "
                f"CoT similarity: `{sim_text}`.\n"
            )
        md.append("\n")
    return "".join(md)


def write_markdown(output_dir, aggregate, summary, detail, examples, anecdotes, figure_paths):
    report_path = output_dir / "IRAC_MULTI_DATASET_REPORT.md"
    total_questions = int(aggregate["questions"].sum())
    total_probes = int(aggregate["probes"].sum())
    total_suir_questions = int(aggregate["questions_with_suir"].sum())
    overall_question_suir = total_suir_questions / total_questions
    total_flip_questions = int(aggregate["questions_with_answer_flip"].sum())
    overall_question_flip = total_flip_questions / total_questions

    detail = detail.copy()
    detail["suir_flag"] = bool_series(detail.get("suir_detected_this_probe", pd.Series(dtype=str)))
    overall_probe_suir = detail["suir_flag"].mean()

    md = []
    md.append("# The Unspoken Logic: Multi-Dataset IRAC Analysis\n")
    md.append("## Executive Summary\n")
    md.append(
        "This report extends the initial IRAC analysis beyond a single dataset by aggregating the existing "
        "GSM8K, StrategyQA, and TruthfulQA outputs in this repository. No new model runs were performed; "
        "the report analyzes the saved experiment CSVs and detailed probe logs.\n"
    )
    md.append(
        f"Across **{total_questions:,} questions** and **{total_probes:,} probes**, "
        f"**{total_suir_questions:,} questions ({overall_question_suir * 100:.1f}%)** had at least one strict SUIR detection. "
        f"The probe-level strict SUIR rate was **{overall_probe_suir * 100:.1f}%**. "
        f"Answer flips were broader: **{total_flip_questions:,} questions ({overall_question_flip * 100:.1f}%)** changed answer under at least one ignore probe.\n"
    )
    md.append(
        "The central pattern from the draft remains visible: answer changes often occur while the generated reasoning remains semantically close to the baseline. "
        "The effect is not confined to arithmetic. StrategyQA exposes missing factual bridges, and TruthfulQA exposes reliance on factual premises and misconception structure.\n"
    )

    md.append("## Dataset Summary\n")
    md.append(
        "| Dataset | Questions | Probes | Questions with SUIR | Mean SUIR rate | Questions with flips | Mean flip rate | Mean CoT sim: ignore | Mean CoT sim: use |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in aggregate.itertuples(index=False):
        md.append(
            f"| {row.dataset} | {row.questions:,} | {row.probes:,} | "
            f"{row.questions_with_suir:,} ({row.question_suir_rate * 100:.1f}%) | "
            f"{row.mean_probe_suir_rate * 100:.1f}% | "
            f"{row.questions_with_answer_flip:,} ({row.question_answer_flip_rate * 100:.1f}%) | "
            f"{row.mean_answer_flip_rate * 100:.1f}% | "
            f"{row.mean_cot_similarity_ignore:.3f} | {row.mean_cot_similarity_use:.3f} |\n"
        )
    md.append("\n")

    for path in figure_paths[:4]:
        md.append(f"![{path.stem.replace('_', ' ').title()}](figures/{path.name})\n\n")

    md.append("## Dataset Findings\n")
    for row in aggregate.itertuples(index=False):
        md.append(f"### {row.dataset}\n")
        if row.dataset == "GSM8K":
            md.append(
                f"GSM8K shows the lowest strict SUIR rate by probe ({row.mean_probe_suir_rate * 100:.1f}%), "
                f"but still has {row.questions_with_suir:,} questions with at least one SUIR detection. "
                f"The high mean baseline-vs-ignore CoT similarity ({row.mean_cot_similarity_ignore:.3f}) means many answer-sensitive probes preserve the same arithmetic explanation template.\n"
            )
        elif row.dataset == "StrategyQA":
            md.append(
                f"StrategyQA has the highest strict SUIR rate ({row.mean_probe_suir_rate * 100:.1f}% by probe) and "
                f"the highest question-level SUIR incidence ({row.question_suir_rate * 100:.1f}%). "
                "The detected factors usually involve missing bridge facts, entity links, or historical/world-knowledge dependencies.\n"
            )
        else:
            md.append(
                f"TruthfulQA has very high answer-flip incidence ({row.question_answer_flip_rate * 100:.1f}% of questions), "
                f"but a lower strict SUIR rate ({row.mean_probe_suir_rate * 100:.1f}% by probe). "
                "This gap is important: many ignore probes perturb the answer without satisfying the stronger baseline/use/ignore contrast criterion.\n"
            )

    md.append(write_anecdote_section(anecdotes))

    md.append("## Category and Example Analysis\n")
    if len(figure_paths) > 4:
        for path in figure_paths[4:]:
            md.append(f"![{path.stem.replace('_', ' ').title()}](figures/{path.name})\n\n")

    md.append("Representative high-similarity SUIR probes:\n\n")
    md.append("| Dataset | Question | Baseline answer | Ignore answer | CoT similarity | Hypothesis |\n")
    md.append("|---|---|---|---|---:|---|\n")
    for row in examples.itertuples(index=False):
        md.append(
            f"| {row.dataset} | {truncate(row.question_text, 110)} | "
            f"{truncate(display_answer(row.baseline_answer), 60)} | {truncate(display_answer(row.answer_ignore), 60)} | "
            f"{float(row.similarity_base_cot_vs_ignore_cot):.3f} | {truncate(row.hypothesis, 115)} |\n"
        )
    md.append("\n")

    md.append("## Interpretation\n")
    md.append(
        "- **SUIR is measurable across task families.** GSM8K primarily surfaces arithmetic and wording shortcuts, while StrategyQA and TruthfulQA surface broader factual and premise dependencies.\n"
        "- **Answer flips alone are too broad.** TruthfulQA flips frequently under ignore probes, but strict SUIR is lower because many flips do not preserve the baseline/use alignment needed for the IRAC criterion.\n"
        "- **CoT invariance remains the strongest warning signal.** High baseline-vs-ignore similarity in detected examples suggests the model often reuses a plausible reasoning frame even when the answer changes.\n"
        "- **The LLM judge validates most generated hypotheses as relevant**, but this should be interpreted cautiously because the hypotheses are themselves model-generated probes.\n"
    )

    md.append("## Limitations\n")
    md.append(
        "- The analysis uses saved outputs only; it does not re-run experiments or estimate sampling variance.\n"
        "- Some answer fields contain parser abbreviations, unknown outputs, or parse errors, especially in yes/no datasets. These are counted in aggregate; unknown outputs and explicit parse errors are filtered out of the example table.\n"
        "- Generated hypotheses can introduce instruction-following artifacts. The report therefore distinguishes answer flips from stricter SUIR detections.\n"
        "- StrategyQA categories are mostly sparse entity labels, so category-level claims are most meaningful for TruthfulQA.\n"
    )

    md.append("## Generated Artifacts\n")
    md.append("- `combined_dataset_summary.csv`: dataset-level aggregate metrics.\n")
    md.append("- `top_suir_examples.csv`: selected high-similarity SUIR examples.\n")
    md.append("- `manual_anecdotal_patterns.csv`: manually curated cross-dataset examples.\n")
    md.append("- `figures/`: cross-dataset charts used in this report.\n")
    md.append("- `IRAC_MULTI_DATASET_REPORT.pdf`: PDF rendering of this report.\n")

    report_path.write_text("".join(md), encoding="utf-8")
    return report_path


def markdown_to_pdf(markdown_path, pdf_path):
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    page_w, page_h = 1700, 2200
    margin = 110
    line_h = 34
    pages = []
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    y = margin

    def add_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = margin

    def ensure_space(height):
        if y + height > page_h - margin:
            add_page()

    for raw_line in lines:
        line = raw_line.rstrip()
        image_match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if image_match:
            image_path = markdown_path.parent / image_match.group(1)
            if image_path.exists():
                with Image.open(image_path) as fig:
                    fig = fig.convert("RGB")
                    max_w = page_w - 2 * margin
                    scale = min(1.0, max_w / fig.width)
                    new_size = (int(fig.width * scale), int(fig.height * scale))
                    fig = fig.resize(new_size)
                    ensure_space(fig.height + 35)
                    page.paste(fig, (margin, int(y)))
                    y += fig.height + 35
            continue

        if not line:
            y += line_h // 2
            continue

        if line.startswith("# "):
            font = load_font(44, bold=True)
            text = line[2:]
            wrapped = textwrap.wrap(text, width=48, break_long_words=False)
            ensure_space(len(wrapped) * 56 + 16)
            for item in wrapped:
                draw.text((margin, y), item, font=font, fill="#1f2933")
                y += 56
            y += 8
            continue
        if line.startswith("## "):
            font = load_font(34, bold=True)
            text = line[3:]
            ensure_space(60)
            draw.text((margin, y + 16), text, font=font, fill="#1f2933")
            y += 68
            continue
        if line.startswith("### "):
            font = load_font(28, bold=True)
            text = line[4:]
            ensure_space(48)
            draw.text((margin, y + 8), text, font=font, fill="#1f2933")
            y += 52
            continue

        font = FONT_MONO if line.startswith("|") else FONT_BODY
        width = 110 if line.startswith("|") else 100
        clean = re.sub(r"\*\*", "", line)
        clean = clean.replace("`", "")
        wrapped = textwrap.wrap(clean, width=width, break_long_words=False) or [""]
        ensure_space(len(wrapped) * line_h + 8)
        for item in wrapped:
            draw.text((margin, y), item, font=font, fill="#1f2933")
            y += line_h
        y += 3

    pages.append(page)
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=150)


def write_validation(output_dir, aggregate, summary, detail, examples, figure_paths):
    validations = {
        "total_questions": int(aggregate["questions"].sum()),
        "total_probes": int(aggregate["probes"].sum()),
        "summary_rows": int(len(summary)),
        "detail_rows": int(len(detail)),
        "figures": len(figure_paths),
        "examples": int(len(examples)),
    }
    lines = [f"{key}: {value}\n" for key, value in validations.items()]
    (output_dir / "VALIDATION_SUMMARY.txt").write_text("".join(lines), encoding="utf-8")
    return validations


def main():
    parser = argparse.ArgumentParser(description="Generate a cross-dataset IRAC analysis report.")
    parser.add_argument("--root", default=".", help="Path to the logic_cot repository root.")
    parser.add_argument("--output-dir", default="combined_dataset_analysis", help="Directory for report artifacts.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF rendering.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundles, summary, detail, aggregate = load_all(root)
    examples = select_examples(detail)
    anecdotes = build_manual_anecdotes(detail)

    aggregate.to_csv(output_dir / "combined_dataset_summary.csv", index=False)
    examples.to_csv(output_dir / "top_suir_examples.csv", index=False)
    anecdotes.to_csv(output_dir / "manual_anecdotal_patterns.csv", index=False)

    figure_paths = [
        figures_dir / "01_cross_dataset_rates.png",
        figures_dir / "02_cot_similarity_by_dataset.png",
        figures_dir / "03_suir_vs_answer_flip.png",
        figures_dir / "04_llm_judge_stacked_bars.png",
        figures_dir / "05_truthfulqa_category_hotspots.png",
        figures_dir / "06_high_similarity_suir_examples.png",
    ]
    plot_cross_dataset_rates(aggregate, figure_paths[0])
    plot_similarity(aggregate, figure_paths[1])
    plot_suir_vs_flip(summary, figure_paths[2])
    plot_judge_stacked(aggregate, figure_paths[3])
    plot_truthfulqa_categories(summary, figure_paths[4])
    plot_examples_table(examples, figure_paths[5])

    report_path = write_markdown(output_dir, aggregate, summary, detail, examples, anecdotes, figure_paths)
    pdf_path = output_dir / "IRAC_MULTI_DATASET_REPORT.pdf"
    if not args.no_pdf:
        markdown_to_pdf(report_path, pdf_path)

    validations = write_validation(output_dir, aggregate, summary, detail, examples, figure_paths)

    print("Generated multi-dataset IRAC report")
    print(f"  Report: {report_path}")
    if not args.no_pdf:
        print(f"  PDF: {pdf_path}")
    print(f"  Figures: {figures_dir}")
    print(f"  Questions: {validations['total_questions']:,}")
    print(f"  Probes: {validations['total_probes']:,}")


if __name__ == "__main__":
    main()
