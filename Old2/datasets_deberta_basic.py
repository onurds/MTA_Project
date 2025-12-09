"""
Compact DeBERTa-based datasets for NER and RE.

Assumes the CSV has:
- header: raw text
- affected_spans: JSON list of route spans with id/start/end
- direction_spans: JSON list of direction spans with id/start/end
- relations: JSON list with route_span_id and direction_span_id

Use with DebertaV2TokenizerFast; it provides offset_mapping so no manual
character alignment is needed beyond simple overlap checks.
"""

from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# Labels
NER_LABEL2ID = {"O": 0, "B-ROUTE": 1, "I-ROUTE": 2, "B-DIRECTION": 3, "I-DIRECTION": 4}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}
IGNORE_INDEX = -100

RE_LABEL2ID = {"NO_RELATION": 0, "HAS_DIRECTION": 1}
RE_ID2LABEL = {v: k for k, v in RE_LABEL2ID.items()}


def parse_json_field(raw) -> List[Dict]:
    if pd.isna(raw) or raw in ("", "[]"):
        return []
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []


def split_df(df: pd.DataFrame, split: Optional[str], train_ratio=0.7, val_ratio=0.15, seed=42):
    if split is None:
        return df
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed)
    val_ratio_adj = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adj, random_state=seed)
    if split == "train":
        return train_df
    if split == "val":
        return val_df
    if split == "test":
        return test_df
    raise ValueError(f"Invalid split: {split}")


class MTANERDatasetDeberta(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 256,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_csv(csv_path)
        self.df = split_df(df, split, train_ratio, val_ratio, random_state).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["header"]) if pd.notna(row["header"]) else ""
        route_spans = parse_json_field(row.get("affected_spans", "[]"))
        direction_spans = parse_json_field(row.get("direction_spans", "[]"))

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        labels = self._build_labels(encoded["offset_mapping"], route_spans, direction_spans)

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_labels(
        self, offsets: List[Tuple[int, int]], route_spans: List[Dict], direction_spans: List[Dict]
    ) -> List[int]:
        labels = [NER_LABEL2ID["O"]] * len(offsets)

        # Mark special tokens / padding
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                labels[i] = IGNORE_INDEX

        def mark_span(spans: List[Dict], b_label: int, i_label: int):
            for span in spans:
                span_start, span_end = span["start"], span["end"]
                first = True
                for i, (tok_start, tok_end) in enumerate(offsets):
                    if labels[i] == IGNORE_INDEX:
                        continue
                    if tok_start < span_end and tok_end > span_start:
                        labels[i] = b_label if first else i_label
                        first = False

        mark_span(route_spans, NER_LABEL2ID["B-ROUTE"], NER_LABEL2ID["I-ROUTE"])
        mark_span(direction_spans, NER_LABEL2ID["B-DIRECTION"], NER_LABEL2ID["I-DIRECTION"])
        return labels


class MTAREDatasetDeberta(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 256,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        include_negatives: bool = True,
        negative_ratio: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio

        df = pd.read_csv(csv_path)
        df = split_df(df, split, train_ratio, val_ratio, random_state)
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        import random

        samples = []
        for _, row in df.iterrows():
            text = str(row["header"]) if pd.notna(row["header"]) else ""
            route_spans = parse_json_field(row.get("affected_spans", "[]"))
            direction_spans = parse_json_field(row.get("direction_spans", "[]"))
            relations = parse_json_field(row.get("relations", "[]"))
            if not route_spans or not direction_spans:
                continue

            positive = {(r["route_span_id"], r["direction_span_id"]) for r in relations}

            for r_span in route_spans:
                for d_span in direction_spans:
                    key = (r_span["id"], d_span["id"])
                    is_pos = key in positive
                    if not is_pos:
                        if not self.include_negatives:
                            continue
                        if self.negative_ratio < 1.0 and random.random() > self.negative_ratio:
                            continue
                    samples.append(
                        {
                            "text": text,
                            "route_spans": route_spans,
                            "direction_spans": direction_spans,
                            "route_id": r_span["id"],
                            "direction_id": d_span["id"],
                            "label": 1 if is_pos else 0,
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        marked = self._insert_markers(
            sample["text"],
            sample["route_spans"],
            sample["direction_spans"],
            sample["route_id"],
            sample["direction_id"],
        )
        encoded = self.tokenizer(
            marked, max_length=self.max_length, padding="max_length", truncation=True
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }

    @staticmethod
    def _insert_markers(
        text: str, route_spans: List[Dict], direction_spans: List[Dict], route_id: int, direction_id: int
    ) -> str:
        r_span = next((s for s in route_spans if s["id"] == route_id), None)
        d_span = next((s for s in direction_spans if s["id"] == direction_id), None)
        if not r_span or not d_span:
            return text

        inserts = [
            (r_span["start"], "[ROUTE] ", False),
            (r_span["end"], " [/ROUTE]", True),
            (d_span["start"], "[DIR] ", False),
            (d_span["end"], " [/DIR]", True),
        ]
        inserts.sort(key=lambda x: (-x[0], x[2]))

        out = text
        for pos, mark, _ in inserts:
            out = out[:pos] + mark + out[pos:]
        return out
