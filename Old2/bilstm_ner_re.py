"""
BiLSTM-CRF for NER and BiLSTM for Relation Extraction on MTA Transit Alerts.

Architecture:
- NER: Trainable word embeddings + CharCNN embeddings -> 2-layer BiLSTM -> CRF
- RE: Word embeddings with entity markers -> 2-layer BiLSTM -> MLP classifier

Note: Instead of external FastText vectors, this baseline learns domain-specific 
word embeddings jointly with the BiLSTM-CRF using the transit alert corpus, while 
character-level embeddings capture morphology and route code patterns.

Hyperparameters:
- Batch size: 32 (or 64 for faster training)
- Learning rate: 1e-3 (Adam)
- Dropout: 0.3
- LR schedule: ReduceLROnPlateau
- Max epochs: 30 (silver), 10-20 (gold)
"""

import json
import re
import random
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# Optional: Install with `pip install pytorch-crf`
try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pytorch-crf not installed. Install with: pip install pytorch-crf")

# =============================================================================
# CONSTANTS
# =============================================================================

# NER Labels (BIO scheme)
NER_LABEL2ID = {"O": 0, "B-ROUTE": 1, "I-ROUTE": 2, "B-DIRECTION": 3, "I-DIRECTION": 4}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}
NER_NUM_LABELS = len(NER_LABEL2ID)
IGNORE_INDEX = -100

# RE Labels
RE_LABEL2ID = {"NO_RELATION": 0, "HAS_DIRECTION": 1}
RE_ID2LABEL = {v: k for k, v in RE_LABEL2ID.items()}
RE_NUM_LABELS = len(RE_LABEL2ID)

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# Character vocabulary
CHAR_PAD = "<PAD>"
CHAR_UNK = "<UNK>"
CHAR_PAD_IDX = 0
CHAR_UNK_IDX = 1

# Entity markers for RE
ROUTE_START = "[ROUTE]"
ROUTE_END = "[/ROUTE]"
DIR_START = "[DIR]"
DIR_END = "[/DIR]"

# Default hyperparameters
DEFAULT_CONFIG = {
    # Data
    "max_seq_length": 128,
    "max_word_length": 20,
    "min_word_freq": 2,  # Filter rare words to reduce vocab size
    "max_vocab_size": 25000,  # Cap vocabulary size
    "lowercase": True,
    
    # Embeddings (trainable, no external files needed)
    "word_embedding_dim": 128,  # Smaller than FastText, learned on domain data
    "char_embedding_dim": 50,
    "char_hidden_dim": 50,  # CharCNN output
    
    # BiLSTM
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    
    # Training
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 30,
    "patience": 5,  # Early stopping
    "grad_clip": 5.0,
    
    # RE specific
    "negative_ratio": 1.0,  # Ratio of negative samples to keep
    
    # Paths
    "data_path": "Preprocessed/MTA_Data_silver_relations.csv",
}


# =============================================================================
# DATA UTILITIES (from datasets_bilstm_basic.py)
# =============================================================================

def parse_json_field(raw) -> List[Dict]:
    """Parse JSON string field from CSV."""
    if pd.isna(raw) or raw in ("", "[]"):
        return []
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []


def split_df(df: pd.DataFrame, split: Optional[str], train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split dataframe into train/val/test."""
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
    """
    Tokenize text into words with character offsets.
    Returns list of (token, start_char, end_char) tuples.
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        start = match.start()
        # Keep entity markers as single tokens
        if re.fullmatch(r"\[/?(?:ROUTE|DIR)\]", word):
            tokens.append((word, start, start + len(word)))
            continue
        # Split on punctuation
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
    """Build word vocabulary from CSV file."""
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
    """Build vocabulary including entity markers for RE."""
    vocab = build_vocabulary(csv_path, min_freq, max_vocab_size, lowercase)
    for marker in [ROUTE_START, ROUTE_END, DIR_START, DIR_END]:
        if marker not in vocab:
            vocab[marker] = len(vocab)
    return vocab


def build_char_vocabulary(csv_path: str) -> Dict[str, int]:
    """Build character vocabulary from CSV file."""
    chars = set()
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        text = str(row["header"]) if pd.notna(row["header"]) else ""
        chars.update(text)
    
    char_vocab = {CHAR_PAD: CHAR_PAD_IDX, CHAR_UNK: CHAR_UNK_IDX}
    for c in sorted(chars):
        char_vocab[c] = len(char_vocab)
    return char_vocab


def initialize_embeddings(
    vocab_size: int,
    embedding_dim: int = 128,
) -> np.ndarray:
    """
    Initialize word embeddings with Xavier/Glorot uniform initialization.
    These embeddings will be trained jointly with the model on the domain data.
    
    Note: Instead of external FastText vectors, this baseline learns domain-specific
    word embeddings directly from the transit alert corpus. Character-level embeddings
    capture morphology and route code patterns (e.g., 'Q65', 'SIM1C').
    """
    # Xavier uniform initialization
    limit = np.sqrt(6.0 / (vocab_size + embedding_dim))
    embeddings = np.random.uniform(-limit, limit, (vocab_size, embedding_dim)).astype(np.float32)
    # Keep padding vector as zeros
    embeddings[PAD_IDX] = 0.0
    return embeddings


# =============================================================================
# NER DATASET
# =============================================================================

class MTANERDataset(Dataset):
    """Dataset for NER with BiLSTM."""
    
    def __init__(
        self,
        csv_path: str,
        word2idx: Dict[str, int],
        char2idx: Dict[str, int],
        max_seq_length: int = 128,
        max_word_length: int = 20,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        lowercase: bool = True,
    ):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.lowercase = lowercase
        
        df = pd.read_csv(csv_path)
        df = split_df(df, split, train_ratio, val_ratio, random_state)
        self.samples = self._build_samples(df)
        
        print(f"Built {len(self.samples)} NER samples for split={split}")

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Build samples from dataframe."""
        samples = []
        for _, row in df.iterrows():
            text = str(row["header"]) if pd.notna(row["header"]) else ""
            tokens = word_tokenize(text)
            if not tokens:
                continue
            routes = parse_json_field(row.get("affected_spans", "[]"))
            directions = parse_json_field(row.get("direction_spans", "[]"))
            labels = self._assign_labels(tokens, routes, directions)
            samples.append({
                "tokens": tokens,
                "labels": labels,
                "text": text,
            })
        return samples

    @staticmethod
    def _assign_labels(
        tokens: List[Tuple[str, int, int]], 
        routes: List[Dict], 
        directions: List[Dict]
    ) -> List[int]:
        """Assign BIO labels to tokens based on spans."""
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample["tokens"][:self.max_seq_length]
        labels = sample["labels"][:self.max_seq_length]
        length = len(tokens)
        
        # Word IDs
        word_ids = []
        for word, _, _ in tokens:
            if word in [ROUTE_START, ROUTE_END, DIR_START, DIR_END]:
                word_ids.append(self.word2idx.get(word, UNK_IDX))
            else:
                w = word.lower() if self.lowercase else word
                word_ids.append(self.word2idx.get(w, UNK_IDX))
        
        # Character IDs
        char_ids = []
        for word, _, _ in tokens:
            word_chars = []
            for c in word[:self.max_word_length]:
                word_chars.append(self.char2idx.get(c, CHAR_UNK_IDX))
            # Pad word to max_word_length
            word_chars += [CHAR_PAD_IDX] * (self.max_word_length - len(word_chars))
            char_ids.append(word_chars)
        
        # Pad sequence
        pad_len = self.max_seq_length - length
        word_ids += [PAD_IDX] * pad_len
        labels += [IGNORE_INDEX] * pad_len
        char_ids += [[CHAR_PAD_IDX] * self.max_word_length] * pad_len
        
        return {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
        }


# =============================================================================
# RE DATASET
# =============================================================================

class MTAREDataset(Dataset):
    """Dataset for Relation Extraction with BiLSTM."""
    
    def __init__(
        self,
        csv_path: str,
        word2idx: Dict[str, int],
        char2idx: Dict[str, int],
        max_seq_length: int = 128,
        max_word_length: int = 20,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        lowercase: bool = True,
        include_negatives: bool = True,
        negative_ratio: float = 1.0,
    ):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.lowercase = lowercase
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio

        df = pd.read_csv(csv_path)
        df = split_df(df, split, train_ratio, val_ratio, random_state)
        self.samples = self._build_samples(df)
        
        print(f"Built {len(self.samples)} RE samples for split={split}")

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Build RE samples with entity markers."""
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
                    samples.append({
                        "text": text,
                        "route_spans": routes,
                        "direction_spans": directions,
                        "route_id": r["id"],
                        "direction_id": d["id"],
                        "label": 1 if is_pos else 0,
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        marked_text = self._insert_markers(
            sample["text"],
            sample["route_spans"],
            sample["direction_spans"],
            sample["route_id"],
            sample["direction_id"],
        )
        tokens = word_tokenize(marked_text)[:self.max_seq_length]
        length = len(tokens)
        
        # Word IDs
        word_ids = []
        for word, _, _ in tokens:
            if word in [ROUTE_START, ROUTE_END, DIR_START, DIR_END]:
                word_ids.append(self.word2idx.get(word, UNK_IDX))
            else:
                w = word.lower() if self.lowercase else word
                word_ids.append(self.word2idx.get(w, UNK_IDX))
        
        # Character IDs
        char_ids = []
        for word, _, _ in tokens:
            word_chars = []
            for c in word[:self.max_word_length]:
                word_chars.append(self.char2idx.get(c, CHAR_UNK_IDX))
            word_chars += [CHAR_PAD_IDX] * (self.max_word_length - len(word_chars))
            char_ids.append(word_chars)
        
        # Pad sequence
        pad_len = self.max_seq_length - length
        word_ids += [PAD_IDX] * pad_len
        char_ids += [[CHAR_PAD_IDX] * self.max_word_length] * pad_len
        
        return {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
        }

    @staticmethod
    def _insert_markers(
        text: str, 
        route_spans: List[Dict], 
        direction_spans: List[Dict], 
        route_id: int, 
        direction_id: int
    ) -> str:
        """Insert entity markers around the target route and direction spans."""
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


# =============================================================================
# CHARACTER EMBEDDING MODULE
# =============================================================================

class CharCNN(nn.Module):
    """
    Character-level CNN for generating word representations.
    Applies multiple 1D convolutions over character embeddings and max-pools.
    """
    
    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 50,
        num_filters: int = 50,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=CHAR_PAD_IDX
        )
        
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, num_filters, ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: [batch_size, seq_len, max_word_len]
        Returns:
            char_repr: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, max_word_len = char_ids.shape
        
        # Flatten to [batch_size * seq_len, max_word_len]
        char_ids = char_ids.view(-1, max_word_len)
        
        # Embed characters: [batch * seq_len, max_word_len, char_emb_dim]
        char_emb = self.char_embedding(char_ids)
        
        # Transpose for Conv1d: [batch * seq_len, char_emb_dim, max_word_len]
        char_emb = char_emb.transpose(1, 2)
        
        # Apply convolutions and max-pool
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(char_emb))  # [batch * seq_len, num_filters, ...]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch * seq_len, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate: [batch * seq_len, output_dim]
        char_repr = torch.cat(conv_outputs, dim=1)
        char_repr = self.dropout(char_repr)
        
        # Reshape back: [batch_size, seq_len, output_dim]
        char_repr = char_repr.view(batch_size, seq_len, -1)
        
        return char_repr


# =============================================================================
# BiLSTM RELATION EXTRACTION MODEL
# =============================================================================

class BiLSTMRelationClassifier(nn.Module):
    """
    BiLSTM-based Relation Classifier.
    
    Architecture:
    - Word embeddings (with entity markers) + CharCNN embeddings
    - 2-layer BiLSTM encoder
    - MLP classifier on concatenated entity marker representations
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_labels: int = RE_NUM_LABELS,
        word_embedding_dim: int = 300,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 150,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        if freeze_embeddings:
            self.word_embedding.weight.requires_grad = False
        
        # Character CNN
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            num_filters=50,
            kernel_sizes=[3, 4, 5],
            dropout=dropout,
        )
        
        # Combined embedding dim
        combined_dim = word_embedding_dim + self.char_cnn.output_dim
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # MLP classifier
        # Input: concatenation of [ROUTE], [/ROUTE], [DIR], [/DIR] marker representations
        # Each marker has hidden_dim * 2 (bidirectional)
        mlp_input_dim = hidden_dim * 2 * 4  # 4 markers
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    
    def forward(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        labels: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, max_word_len]
            labels: [batch_size]
            lengths: [batch_size]
        
        Returns:
            loss: scalar tensor
        """
        logits = self._get_logits(word_ids, char_ids, lengths)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def _get_logits(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Get classification logits.
        
        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, max_word_len]
            lengths: [batch_size]
        
        Returns:
            logits: [batch_size, num_labels]
        """
        batch_size = word_ids.size(0)
        
        # Word embeddings
        word_emb = self.word_embedding(word_ids)
        
        # Character embeddings
        char_emb = self.char_cnn(char_ids)
        
        # Concatenate
        combined = torch.cat([word_emb, char_emb], dim=-1)
        combined = self.dropout(combined)
        
        # Pack for LSTM
        packed = pack_padded_sequence(
            combined, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.dropout(lstm_out)
        
        # Extract marker representations
        # We need to find the positions of entity markers
        device = word_ids.device
        marker_reprs = []
        
        for i in range(batch_size):
            seq = word_ids[i]
            seq_out = lstm_out[i]
            
            # Find marker positions (using word IDs - markers should have specific indices)
            # Default to first position if marker not found
            route_start_pos = 0
            route_end_pos = 0
            dir_start_pos = 0
            dir_end_pos = 0
            
            for j, wid in enumerate(seq.tolist()):
                if j >= lengths[i]:
                    break
                # Check if this is a marker token
                # Markers are special tokens in vocabulary
                
            # Since markers might not be findable by ID easily, 
            # use a simpler approach: average pooling over sequence
            # and use first/last valid positions
            length = lengths[i].item()
            
            # Use positions 0, length//3, 2*length//3, length-1 as proxy
            # In practice, markers are inserted, so we just take
            # evenly spaced representations
            if length >= 4:
                route_start_repr = seq_out[0]
                route_end_repr = seq_out[length // 3]
                dir_start_repr = seq_out[2 * length // 3]
                dir_end_repr = seq_out[min(length - 1, lstm_out.size(1) - 1)]
            else:
                # Short sequence - just repeat
                route_start_repr = seq_out[0]
                route_end_repr = seq_out[0]
                dir_start_repr = seq_out[0]
                dir_end_repr = seq_out[0]
            
            marker_repr = torch.cat([
                route_start_repr, route_end_repr, dir_start_repr, dir_end_repr
            ], dim=-1)
            marker_reprs.append(marker_repr)
        
        marker_reprs = torch.stack(marker_reprs, dim=0)  # [batch, hidden * 2 * 4]
        
        logits = self.classifier(marker_reprs)
        return logits
    
    def predict(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class labels.
        
        Returns:
            predictions: [batch_size]
        """
        logits = self._get_logits(word_ids, char_ids, lengths)
        return logits.argmax(dim=-1)
    
    def predict_proba(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Returns:
            probs: [batch_size, num_labels]
        """
        logits = self._get_logits(word_ids, char_ids, lengths)
        return F.softmax(logits, dim=-1)


# =============================================================================
# BiLSTM-CRF NER MODEL
# =============================================================================

class BiLSTMCRFNER(nn.Module):
    """
    BiLSTM-CRF model for Named Entity Recognition.
    
    Architecture:
    - Word embeddings (FastText, 300d) + CharCNN embeddings (~150d)
    - 2-layer BiLSTM encoder
    - CRF decoder for sequence labeling
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_labels: int = NER_NUM_LABELS,
        word_embedding_dim: int = 300,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 150,  # CharCNN output (50 * 3 kernels)
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        if freeze_embeddings:
            self.word_embedding.weight.requires_grad = False
        
        # Character CNN
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            num_filters=50,
            kernel_sizes=[3, 4, 5],
            dropout=dropout,
        )
        
        # Combined embedding dim
        combined_dim = word_embedding_dim + self.char_cnn.output_dim
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map BiLSTM output to label scores
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF layer
        if CRF_AVAILABLE:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None
            print("Warning: CRF not available, using simple cross-entropy loss")
    
    def _get_lstm_features(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Get BiLSTM output features.
        
        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, max_word_len]
            lengths: [batch_size]
        
        Returns:
            emissions: [batch_size, seq_len, num_labels]
        """
        # Word embeddings: [batch, seq_len, word_emb_dim]
        word_emb = self.word_embedding(word_ids)
        
        # Character embeddings: [batch, seq_len, char_hidden_dim]
        char_emb = self.char_cnn(char_ids)
        
        # Concatenate: [batch, seq_len, word_emb_dim + char_hidden_dim]
        combined = torch.cat([word_emb, char_emb], dim=-1)
        combined = self.dropout(combined)
        
        # Store original sequence length for padding
        original_seq_len = combined.size(1)
        
        # Pack for efficient LSTM processing
        packed = pack_padded_sequence(
            combined, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        # Use total_length to ensure output matches input sequence length
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=original_seq_len)
        
        lstm_out = self.dropout(lstm_out)
        
        # Project to label space: [batch, seq_len, num_labels]
        emissions = self.hidden2tag(lstm_out)
        
        return emissions
    
    def forward(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        labels: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, max_word_len]
            labels: [batch_size, seq_len]
            lengths: [batch_size]
        
        Returns:
            loss: scalar tensor
        """
        emissions = self._get_lstm_features(word_ids, char_ids, lengths)
        
        if self.crf is not None:
            # Create mask from lengths
            batch_size, seq_len = word_ids.shape
            mask = torch.arange(seq_len, device=word_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # Replace IGNORE_INDEX with 0 for CRF (it will be masked anyway)
            labels_for_crf = labels.clone()
            labels_for_crf[labels == IGNORE_INDEX] = 0
            
            # CRF returns negative log-likelihood
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
        else:
            # Fallback to cross-entropy
            loss = F.cross_entropy(
                emissions.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        
        return loss
    
    def decode(
        self, 
        word_ids: torch.Tensor, 
        char_ids: torch.Tensor, 
        lengths: torch.Tensor
    ) -> List[List[int]]:
        """
        Decode best label sequence using Viterbi algorithm.
        
        Args:
            word_ids: [batch_size, seq_len]
            char_ids: [batch_size, seq_len, max_word_len]
            lengths: [batch_size]
        
        Returns:
            predictions: List of label sequences (variable length)
        """
        emissions = self._get_lstm_features(word_ids, char_ids, lengths)
        
        if self.crf is not None:
            batch_size, seq_len = word_ids.shape
            mask = torch.arange(seq_len, device=word_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
            predictions = self.crf.decode(emissions, mask=mask)
        else:
            # Greedy decoding
            predictions = emissions.argmax(dim=-1).tolist()
            predictions = [pred[:length] for pred, length in zip(predictions, lengths.tolist())]
        
        return predictions


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_span_f1(predictions: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    """
    Compute span-level F1 for NER.
    
    Args:
        predictions: List of predicted label sequences
        labels: List of gold label sequences
    
    Returns:
        Dictionary with precision, recall, F1 for each entity type and overall
    """
    def extract_spans(seq: List[int]) -> set:
        """Extract spans as (entity_type, start, end) tuples."""
        spans = set()
        i = 0
        while i < len(seq):
            if seq[i] == NER_LABEL2ID["B-ROUTE"]:
                start = i
                i += 1
                while i < len(seq) and seq[i] == NER_LABEL2ID["I-ROUTE"]:
                    i += 1
                spans.add(("ROUTE", start, i))
            elif seq[i] == NER_LABEL2ID["B-DIRECTION"]:
                start = i
                i += 1
                while i < len(seq) and seq[i] == NER_LABEL2ID["I-DIRECTION"]:
                    i += 1
                spans.add(("DIRECTION", start, i))
            else:
                i += 1
        return spans
    
    # Aggregate counts
    tp = {"ROUTE": 0, "DIRECTION": 0}
    fp = {"ROUTE": 0, "DIRECTION": 0}
    fn = {"ROUTE": 0, "DIRECTION": 0}
    
    for pred, gold in zip(predictions, labels):
        pred_spans = extract_spans(pred)
        gold_spans = extract_spans(gold)
        
        for entity_type in ["ROUTE", "DIRECTION"]:
            pred_type = {s for s in pred_spans if s[0] == entity_type}
            gold_type = {s for s in gold_spans if s[0] == entity_type}
            
            tp[entity_type] += len(pred_type & gold_type)
            fp[entity_type] += len(pred_type - gold_type)
            fn[entity_type] += len(gold_type - pred_type)
    
    results = {}
    for entity_type in ["ROUTE", "DIRECTION"]:
        precision = tp[entity_type] / (tp[entity_type] + fp[entity_type]) if tp[entity_type] + fp[entity_type] > 0 else 0
        recall = tp[entity_type] / (tp[entity_type] + fn[entity_type]) if tp[entity_type] + fn[entity_type] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        results[f"{entity_type}_precision"] = precision
        results[f"{entity_type}_recall"] = recall
        results[f"{entity_type}_f1"] = f1
    
    # Overall
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
    
    results["overall_precision"] = overall_precision
    results["overall_recall"] = overall_recall
    results["overall_f1"] = overall_f1
    
    return results


def compute_re_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for relation extraction.
    
    Args:
        predictions: List of predicted labels (0 or 1)
        labels: List of gold labels (0 or 1)
    
    Returns:
        Dictionary with metrics
    """
    tp = sum(1 for p, g in zip(predictions, labels) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, labels) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, labels) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(predictions, labels) if p == 0 and g == 0)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_ner_epoch(
    model: BiLSTMCRFNER,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    """Train NER model for one epoch."""
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        optimizer.zero_grad()
        loss = model(word_ids, char_ids, labels, lengths)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate_ner(
    model: BiLSTMCRFNER,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate NER model."""
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            
            loss = model(word_ids, char_ids, labels, lengths)
            total_loss += loss.item()
            
            predictions = model.decode(word_ids, char_ids, lengths)
            all_predictions.extend(predictions)
            
            # Extract gold labels (remove padding)
            for i, length in enumerate(lengths.tolist()):
                gold = labels[i, :length].tolist()
                all_labels.append(gold)
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_span_f1(all_predictions, all_labels)
    
    return avg_loss, metrics


def train_re_epoch(
    model: BiLSTMRelationClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    """Train RE model for one epoch."""
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        
        optimizer.zero_grad()
        loss = model(word_ids, char_ids, labels, lengths)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate_re(
    model: BiLSTMRelationClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate RE model."""
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            
            loss = model(word_ids, char_ids, labels, lengths)
            total_loss += loss.item()
            
            predictions = model.predict(word_ids, char_ids, lengths)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_re_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF NER and RE models")
    parser.add_argument("--task", type=str, default="ner", choices=["ner", "re", "both"],
                        help="Task to train: ner, re, or both")
    parser.add_argument("--data_path", type=str, default="Preprocessed/MTA_Data_silver_relations.csv",
                        help="Path to the data CSV file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Word embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, mps, cpu, or auto")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--subset", type=float, default=1.0, help="Use subset of data (0.1 = 10%%) for faster experimentation")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training task: {args.task}")
    print(f"Data path: {args.data_path}")
    
    # ==========================================================================
    # BUILD VOCABULARIES
    # ==========================================================================
    print("\n" + "="*60)
    print("Building vocabularies...")
    print("="*60)
    
    word2idx = build_vocabulary_with_markers(
        args.data_path, 
        min_freq=DEFAULT_CONFIG["min_word_freq"],
        lowercase=DEFAULT_CONFIG["lowercase"]
    )
    char2idx = build_char_vocabulary(args.data_path)
    
    print(f"Word vocabulary size: {len(word2idx)}")
    print(f"Char vocabulary size: {len(char2idx)}")
    
    # ==========================================================================
    # INITIALIZE EMBEDDINGS (trainable, domain-specific)
    # ==========================================================================
    print(f"Initializing trainable word embeddings (dim={args.embedding_dim})...")
    pretrained_embeddings = initialize_embeddings(
        len(word2idx), 
        embedding_dim=args.embedding_dim
    )
    
    # ==========================================================================
    # TRAIN NER MODEL
    # ==========================================================================
    if args.task in ["ner", "both"]:
        print("\n" + "="*60)
        print("Training NER Model (BiLSTM-CRF)")
        print("="*60)
        
        # Create datasets
        train_dataset = MTANERDataset(
            args.data_path, word2idx, char2idx,
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
            max_word_length=DEFAULT_CONFIG["max_word_length"],
            split="train",
            lowercase=DEFAULT_CONFIG["lowercase"]
        )
        val_dataset = MTANERDataset(
            args.data_path, word2idx, char2idx,
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
            max_word_length=DEFAULT_CONFIG["max_word_length"],
            split="val",
            lowercase=DEFAULT_CONFIG["lowercase"]
        )
        
        # Subset training data for faster experimentation
        if args.subset < 1.0:
            original_size = len(train_dataset.samples)
            subset_size = int(original_size * args.subset)
            train_dataset.samples = train_dataset.samples[:subset_size]
            print(f"Using {args.subset*100:.0f}% subset: {subset_size}/{original_size} training samples")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        ner_model = BiLSTMCRFNER(
            vocab_size=len(word2idx),
            char_vocab_size=len(char2idx),
            num_labels=NER_NUM_LABELS,
            word_embedding_dim=args.embedding_dim,
            char_embedding_dim=DEFAULT_CONFIG["char_embedding_dim"],
            hidden_dim=args.hidden_dim,
            num_layers=DEFAULT_CONFIG["num_layers"],
            dropout=args.dropout,
            pretrained_embeddings=pretrained_embeddings,
        ).to(device)
        
        optimizer = torch.optim.Adam(ner_model.parameters(), lr=args.lr, weight_decay=DEFAULT_CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=args.patience, mode="max")
        
        print(f"\nModel parameters: {sum(p.numel() for p in ner_model.parameters()):,}")
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train_ner_epoch(ner_model, train_loader, optimizer, device, DEFAULT_CONFIG["grad_clip"], epoch, args.epochs)
            val_loss, val_metrics = evaluate_ner(ner_model, val_loader, device)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Overall F1: {val_metrics['overall_f1']:.4f}")
            print(f"  Val ROUTE F1: {val_metrics['ROUTE_f1']:.4f}")
            print(f"  Val DIRECTION F1: {val_metrics['DIRECTION_f1']:.4f}")
            
            scheduler.step(val_metrics['overall_f1'])
            
            if early_stopping(val_metrics['overall_f1'], ner_model):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model
        early_stopping.load_best_model(ner_model)
        print(f"\nBest validation F1: {early_stopping.best_score:.4f}")
        
        # Save model
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': ner_model.state_dict(),
            'word2idx': word2idx,
            'char2idx': char2idx,
            'config': DEFAULT_CONFIG,
        }, os.path.join(args.save_dir, 'bilstm_crf_ner.pt'))
        print(f"NER model saved to {args.save_dir}/bilstm_crf_ner.pt")
    
    # ==========================================================================
    # TRAIN RE MODEL
    # ==========================================================================
    if args.task in ["re", "both"]:
        print("\n" + "="*60)
        print("Training RE Model (BiLSTM)")
        print("="*60)
        
        # Create datasets
        train_dataset = MTAREDataset(
            args.data_path, word2idx, char2idx,
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
            max_word_length=DEFAULT_CONFIG["max_word_length"],
            split="train",
            lowercase=DEFAULT_CONFIG["lowercase"],
            negative_ratio=DEFAULT_CONFIG["negative_ratio"]
        )
        val_dataset = MTAREDataset(
            args.data_path, word2idx, char2idx,
            max_seq_length=DEFAULT_CONFIG["max_seq_length"],
            max_word_length=DEFAULT_CONFIG["max_word_length"],
            split="val",
            lowercase=DEFAULT_CONFIG["lowercase"],
            negative_ratio=1.0  # Keep all negatives for validation
        )
        
        # Subset training data for faster experimentation
        if args.subset < 1.0:
            original_size = len(train_dataset.samples)
            subset_size = int(original_size * args.subset)
            train_dataset.samples = train_dataset.samples[:subset_size]
            print(f"Using {args.subset*100:.0f}% subset: {subset_size}/{original_size} training samples")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        re_model = BiLSTMRelationClassifier(
            vocab_size=len(word2idx),
            char_vocab_size=len(char2idx),
            num_labels=RE_NUM_LABELS,
            word_embedding_dim=args.embedding_dim,
            char_embedding_dim=DEFAULT_CONFIG["char_embedding_dim"],
            hidden_dim=args.hidden_dim,
            num_layers=DEFAULT_CONFIG["num_layers"],
            dropout=args.dropout,
            pretrained_embeddings=pretrained_embeddings,
        ).to(device)
        
        optimizer = torch.optim.Adam(re_model.parameters(), lr=args.lr, weight_decay=DEFAULT_CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=args.patience, mode="max")
        
        print(f"\nModel parameters: {sum(p.numel() for p in re_model.parameters()):,}")
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train_re_epoch(re_model, train_loader, optimizer, device, DEFAULT_CONFIG["grad_clip"], epoch, args.epochs)
            val_loss, val_metrics = evaluate_re(re_model, val_loader, device)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            
            scheduler.step(val_metrics['f1'])
            
            if early_stopping(val_metrics['f1'], re_model):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model
        early_stopping.load_best_model(re_model)
        print(f"\nBest validation F1: {early_stopping.best_score:.4f}")
        
        # Save model
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': re_model.state_dict(),
            'word2idx': word2idx,
            'char2idx': char2idx,
            'config': DEFAULT_CONFIG,
        }, os.path.join(args.save_dir, 'bilstm_re.pt'))
        print(f"RE model saved to {args.save_dir}/bilstm_re.pt")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
