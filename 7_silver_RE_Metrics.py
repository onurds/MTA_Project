import argparse
import json
from collections import Counter

import pandas as pd


def parse_json_list(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def multiset_accuracy(gold_items, silver_items):
    gold_counts = Counter(gold_items)
    silver_counts = Counter(silver_items)
    correct = sum((gold_counts & silver_counts).values())
    total_gold = sum(gold_counts.values())
    total_silver = sum(silver_counts.values())
    if total_gold == 0:
        accuracy = 1.0 if total_silver == 0 else 0.0
    else:
        accuracy = correct / total_gold
    exact_match = gold_counts == silver_counts
    return correct, total_gold, total_silver, accuracy, exact_match


def span_key(span):
    return (
        span.get("start"),
        span.get("end"),
        span.get("type"),
        span.get("value"),
    )


def relation_name_key(rel):
    return (
        rel.get("route"),
        rel.get("direction"),
    )


def relation_span_key(rel):
    return (
        rel.get("route_span_id"),
        rel.get("direction_span_id"),
        rel.get("type"),
    )


def evaluate(df):
    metrics = {
        "relation_labels": {"correct": 0, "gold": 0, "rows_exact": 0},
        "relation_spans": {"correct": 0, "gold": 0, "rows_exact": 0},
        "affected_spans": {"correct": 0, "gold": 0, "rows_exact": 0},
        "direction_spans": {"correct": 0, "gold": 0, "rows_exact": 0},
    }

    for _, row in df.iterrows():
        # Relation label accuracy (route+direction pairs)
        gold_rel_names = [relation_name_key(r) for r in parse_json_list(row["relation_names_gold"])]
        silver_rel_names = [relation_name_key(r) for r in parse_json_list(row["relation_names_silver"])]
        correct, total_gold, total_silver, _, exact = multiset_accuracy(
            gold_rel_names, silver_rel_names
        )
        metrics["relation_labels"]["correct"] += correct
        metrics["relation_labels"]["gold"] += total_gold
        metrics["relation_labels"]["silver"] = metrics["relation_labels"].get("silver", 0) + total_silver
        metrics["relation_labels"]["rows_exact"] += int(exact)

        # Relation spans accuracy (route_span_id + direction_span_id + type)
        gold_rel_spans = [relation_span_key(r) for r in parse_json_list(row["relations_gold"])]
        silver_rel_spans = [relation_span_key(r) for r in parse_json_list(row["relations_silver"])]
        correct, total_gold, total_silver, _, exact = multiset_accuracy(
            gold_rel_spans, silver_rel_spans
        )
        metrics["relation_spans"]["correct"] += correct
        metrics["relation_spans"]["gold"] += total_gold
        metrics["relation_spans"]["silver"] = metrics["relation_spans"].get("silver", 0) + total_silver
        metrics["relation_spans"]["rows_exact"] += int(exact)

        # Affected spans accuracy (route spans)
        gold_aff = [span_key(s) for s in parse_json_list(row["affected_spans_gold"])]
        silver_aff = [span_key(s) for s in parse_json_list(row["affected_spans_silver"])]
        correct, total_gold, total_silver, _, exact = multiset_accuracy(gold_aff, silver_aff)
        metrics["affected_spans"]["correct"] += correct
        metrics["affected_spans"]["gold"] += total_gold
        metrics["affected_spans"]["silver"] = metrics["affected_spans"].get("silver", 0) + total_silver
        metrics["affected_spans"]["rows_exact"] += int(exact)

        # Direction spans accuracy
        gold_dir = [span_key(s) for s in parse_json_list(row["direction_spans_gold"])]
        silver_dir = [span_key(s) for s in parse_json_list(row["direction_spans_silver"])]
        correct, total_gold, total_silver, _, exact = multiset_accuracy(gold_dir, silver_dir)
        metrics["direction_spans"]["correct"] += correct
        metrics["direction_spans"]["gold"] += total_gold
        metrics["direction_spans"]["silver"] = metrics["direction_spans"].get("silver", 0) + total_silver
        metrics["direction_spans"]["rows_exact"] += int(exact)

    return metrics


def format_results(metrics, total_rows):
    lines = []
    rows = []
    for key in ["relation_labels", "relation_spans", "affected_spans", "direction_spans"]:
        correct = metrics[key]["correct"]
        gold = metrics[key]["gold"]
        silver = metrics[key].get("silver", 0)
        rows_exact = metrics[key]["rows_exact"]
        micro_acc = (correct / gold) if gold else 1.0
        precision = (correct / silver) if silver else 1.0
        recall = (correct / gold) if gold else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        row_acc = rows_exact / total_rows if total_rows else 0.0
        lines.append(
            f"{key}: micro-accuracy={micro_acc:.4f} (correct={correct}, gold={gold}), "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, "
            f"row-exact-accuracy={row_acc:.4f} (exact={rows_exact}, rows={total_rows})"
        )
        rows.append(
            {
                "metric": key,
                "micro_accuracy": micro_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "row_exact_accuracy": row_acc,
                "correct": correct,
                "gold": gold,
                "silver": silver,
                "rows_exact": rows_exact,
                "rows_total": total_rows,
            }
        )
    return "\n".join(lines), rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute silver vs gold accuracy for relation labels and spans."
    )
    parser.add_argument(
        "--input",
        default="Preprocessed/MTA_Data_Final_Gold.csv",
    )
    parser.add_argument(
        "--output",
        default="Results/Silver_RE_Metrics.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metrics = evaluate(df)
    text, rows = format_results(metrics, len(df))
    print(text)
    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
