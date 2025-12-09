"""
Compact BiLSTM-oriented datasets for NER and RE.

Assumes the CSV has:
- header: raw text
- affected_spans: JSON list of route spans with id/start/end
- direction_spans: JSON list of direction spans with id/start/end
- relations: JSON list with route_span_id and direction_span_id
"""

from typing import Dict, List, Optional, Tuple
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


NER_LABEL2ID = {"O": 0, "B-ROUTE": 1, "I-ROUTE": 2, "B-DIRECTION": 3, "I-DIRECTION": 4}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}
IGNORE_INDEX = -100

RE_LABEL2ID = {"NO_RELATION": 0, "HAS_DIRECTION": 1}
RE_ID2LABEL = {v: k for k, v in RE_LABEL2ID.items()}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

ROUTE_START = "[ROUTE]"
ROUTE_END = "[/ROUTE]"
DIR_START = "[DIR]"
DIR_END = "[/DIR]"


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


def word_tokenize(text: str) -> List[Tuple[str, int, int]]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        start = match.start()
        if re.fullmatch(r"\[/?(?:ROUTE|DIR)\]", word):
            tokens.append((word, start, start + len(word)))
            continue
        for sub in re.finditer(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", word):
            s = start + sub.start()
            e = s + len(sub.group())
            tokens.append((sub.group(), s, e))
    return tokens


def build_vocabulary(
    csv_path: str,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    lowercase: bool = True,
) -> Dict[str, int]:
    from collections import Counter

    counter = Counter()
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        text = str(row["header"]) if pd.notna(row["header"]) else ""
        for word, _, _ in word_tokenize(text):
            w = word.lower() if lowercase else word
            counter[w] += 1

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])
    if max_vocab_size:
        items = items[: max_vocab_size - len(vocab)]
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def build_vocabulary_with_markers(
    csv_path: str, min_freq: int = 1, max_vocab_size: Optional[int] = None, lowercase: bool = True
) -> Dict[str, int]:
    vocab = build_vocabulary(csv_path, min_freq, max_vocab_size, lowercase)
    for marker in [ROUTE_START, ROUTE_END, DIR_START, DIR_END]:
        if marker not in vocab:
            vocab[marker] = len(vocab)
    return vocab


def load_pretrained_embeddings(
    word2idx: Dict[str, int], embeddings_path: str, embedding_dim: int = 300, lowercase: bool = True
) -> np.ndarray:
    embeddings = np.random.randn(len(word2idx), embedding_dim) * 0.1
    embeddings[PAD_IDX] = 0
    with open(embeddings_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < embedding_dim + 1:
                continue
            word = parts[0].lower() if lowercase else parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.asarray(parts[1:], dtype=float)
    return embeddings


class MTANERDatasetBiLSTM(Dataset):
    def __init__(
        self,
        csv_path: str,
        word2idx: Dict[str, int],
        max_length: int = 128,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        lowercase: bool = True,
    ):
        self.word2idx = word2idx
        self.max_length = max_length
        self.lowercase = lowercase
        df = pd.read_csv(csv_path)
        df = split_df(df, split, train_ratio, val_ratio, random_state)
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        samples = []
        for _, row in df.iterrows():
            text = str(row["header"]) if pd.notna(row["header"]) else ""
            tokens = word_tokenize(text)
            if not tokens:
                continue
            routes = parse_json_field(row.get("affected_spans", "[]"))
            directions = parse_json_field(row.get("direction_spans", "[]"))
            labels = self._assign_labels(tokens, routes, directions)
            samples.append({"tokens": tokens, "labels": labels})
        return samples

    @staticmethod
    def _assign_labels(tokens: List[Tuple[str, int, int]], routes: List[Dict], directions: List[Dict]) -> List[int]:
        labels = [NER_LABEL2ID["O"]] * len(tokens)

        def mark(spans: List[Dict], b_label: int, i_label: int):
            for span in spans:
                start, end = span["start"], span["end"]
                inside = False
                for i, (_, s, e) in enumerate(tokens):
                    if s >= start and e <= end:
                        labels[i] = b_label if not inside else i_label
                        inside = True
                    elif e > end:
                        break

        mark(routes, NER_LABEL2ID["B-ROUTE"], NER_LABEL2ID["I-ROUTE"])
        mark(directions, NER_LABEL2ID["B-DIRECTION"], NER_LABEL2ID["I-DIRECTION"])
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        token_ids = []
        for word, _, _ in sample["tokens"][: self.max_length]:
            w = word if word in [ROUTE_START, ROUTE_END, DIR_START, DIR_END] else (
                word.lower() if self.lowercase else word
            )
            token_ids.append(self.word2idx.get(w, UNK_IDX))

        labels = sample["labels"][: self.max_length]
        length = len(token_ids)

        pad = self.max_length - length
        token_ids += [PAD_IDX] * pad
        labels += [IGNORE_INDEX] * pad

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
        }


class MTAREDatasetBiLSTM(Dataset):
    def __init__(
        self,
        csv_path: str,
        word2idx: Dict[str, int],
        max_length: int = 128,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        lowercase: bool = True,
        include_negatives: bool = True,
        negative_ratio: float = 1.0,
    ):
        self.word2idx = word2idx
        self.max_length = max_length
        self.lowercase = lowercase
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
            routes = parse_json_field(row.get("affected_spans", "[]"))
            directions = parse_json_field(row.get("direction_spans", "[]"))
            relations = parse_json_field(row.get("relations", "[]"))
            if not routes or not directions:
                continue
            positives = {(r["route_span_id"], r["direction_span_id"]) for r in relations}

            for r in routes:
                for d in directions:
                    key = (r["id"], d["id"])
                    is_pos = key in positives
                    if not is_pos:
                        if not self.include_negatives:
                            continue
                        if self.negative_ratio < 1.0 and random.random() > self.negative_ratio:
                            continue
                    samples.append(
                        {
                            "text": text,
                            "route_spans": routes,
                            "direction_spans": directions,
                            "route_id": r["id"],
                            "direction_id": d["id"],
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
        tokens = word_tokenize(marked)
        token_ids = []
        for word, _, _ in tokens[: self.max_length]:
            if word in [ROUTE_START, ROUTE_END, DIR_START, DIR_END]:
                token_ids.append(self.word2idx.get(word, UNK_IDX))
            else:
                w = word.lower() if self.lowercase else word
                token_ids.append(self.word2idx.get(w, UNK_IDX))

        length = len(token_ids)
        pad = self.max_length - length
        token_ids += [PAD_IDX] * pad

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
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
            (r_span["start"], ROUTE_START + " ", False),
            (r_span["end"], " " + ROUTE_END, True),
            (d_span["start"], DIR_START + " ", False),
            (d_span["end"], " " + DIR_END, True),
        ]
        inserts.sort(key=lambda x: (-x[0], x[2]))
        out = text
        for pos, mark, _ in inserts:
            out = out[:pos] + mark + out[pos:]
        return out
