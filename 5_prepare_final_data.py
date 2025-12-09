"""
Prepare Final Data for NER and RE Training

This script converts character-level span annotations to:
1. NER: Token-level BIO labels aligned with DeBERTa tokenizer
2. RE: Text with entity markers + relation labels

Input: Preprocessed/MTA_Data_silver_relations.csv
Output: 
  - final_data/ner_dataset (HuggingFace Dataset)
  - final_data/re_dataset (HuggingFace Dataset)

BIO Label Scheme:
  0: O (Outside)
  1: B-ROUTE (Beginning of route)
  2: I-ROUTE (Inside route)
  3: B-DIRECTION (Beginning of direction)
  4: I-DIRECTION (Inside direction)
  -100: Special tokens (ignored by loss)
"""

import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datasets import Dataset, DatasetDict
from transformers import DebertaV2Tokenizer
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
INPUT_FILE = "Preprocessed/MTA_Data_silver_relations.csv"
OUTPUT_DIR = "final_data"

# BIO Label mapping
LABEL2ID = {
    "O": 0,
    "B-ROUTE": 1,
    "I-ROUTE": 2,
    "B-DIRECTION": 3,
    "I-DIRECTION": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
IGNORE_INDEX = -100


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the silver relations dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df


def parse_json_field(json_str: str) -> List[Dict]:
    """Parse JSON string field to list of dicts."""
    if pd.isna(json_str) or json_str == '[]' or json_str == '':
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


# ============================================================================
# NER: CHARACTER TO TOKEN ALIGNMENT
# ============================================================================

def compute_token_offsets(text: str, tokenizer) -> List[Tuple[int, int]]:
    """
    Compute character offsets for each token manually.
    
    This is needed because the slow tokenizer doesn't support return_offsets_mapping.
    We decode each token and find its position in the original text.
    """
    tokens = tokenizer.tokenize(text)
    offsets = []
    current_pos = 0
    
    for token in tokens:
        # DeBERTa uses ▁ (U+2581) to mark word boundaries
        # Clean the token to get the actual text
        clean_token = token.replace('▁', ' ').strip()
        
        if not clean_token:
            # Handle special boundary markers
            offsets.append((current_pos, current_pos))
            continue
        
        # Find the token in the remaining text (case-insensitive for robustness)
        remaining_text = text[current_pos:]
        
        # Try exact match first
        idx = remaining_text.find(clean_token)
        
        if idx == -1:
            # Try case-insensitive
            idx = remaining_text.lower().find(clean_token.lower())
        
        if idx == -1:
            # Token not found, skip (shouldn't happen often)
            offsets.append((current_pos, current_pos))
        else:
            start = current_pos + idx
            end = start + len(clean_token)
            offsets.append((start, end))
            current_pos = end
    
    return offsets


def align_spans_to_tokens(
    text: str,
    route_spans: List[Dict],
    direction_spans: List[Dict],
    tokenizer,
    max_length: int = MAX_LENGTH
) -> Dict:
    """
    Convert character-level spans to token-level BIO labels.
    
    Args:
        text: The header text
        route_spans: List of route span dicts with 'start', 'end', 'type', 'value'
        direction_spans: List of direction span dicts
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dict with input_ids, attention_mask, labels (BIO tags)
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Initialize all labels as O (outside)
    labels = [LABEL2ID["O"]] * len(input_ids)
    
    # Get token offsets manually
    token_offsets = compute_token_offsets(text, tokenizer)
    
    # Account for [CLS] token at the start
    # DeBERTa adds [CLS] at position 0
    offset_mapping = [(0, 0)]  # [CLS] token
    offset_mapping.extend(token_offsets)
    
    # Pad to match input_ids length (for [SEP] and padding)
    while len(offset_mapping) < len(input_ids):
        offset_mapping.append((0, 0))
    
    # Truncate if needed
    offset_mapping = offset_mapping[:len(input_ids)]
    
    # Mark special tokens with IGNORE_INDEX
    # First token is [CLS], mark as ignore
    labels[0] = IGNORE_INDEX
    
    # Find [SEP] and padding tokens
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    for i, token_id in enumerate(input_ids):
        if token_id == sep_token_id or token_id == pad_token_id:
            labels[i] = IGNORE_INDEX
    
    # Process ROUTE spans
    for span in route_spans:
        span_start = span['start']
        span_end = span['end']
        is_first_token = True
        
        for i, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens
            if labels[i] == IGNORE_INDEX:
                continue
            if token_start == 0 and token_end == 0:
                continue
            
            # Check if token overlaps with span
            if token_start < span_end and token_end > span_start:
                if is_first_token:
                    labels[i] = LABEL2ID["B-ROUTE"]
                    is_first_token = False
                else:
                    labels[i] = LABEL2ID["I-ROUTE"]
    
    # Process DIRECTION spans
    for span in direction_spans:
        span_start = span['start']
        span_end = span['end']
        is_first_token = True
        
        for i, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens
            if labels[i] == IGNORE_INDEX:
                continue
            if token_start == 0 and token_end == 0:
                continue
            
            # Check if token overlaps with span
            if token_start < span_end and token_end > span_start:
                if is_first_token:
                    labels[i] = LABEL2ID["B-DIRECTION"]
                    is_first_token = False
                else:
                    labels[i] = LABEL2ID["I-DIRECTION"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_ner_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """
    Prepare the full NER dataset with token-aligned BIO labels.
    """
    print("\nPreparing NER dataset...")
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {total:,} records...")
        
        text = str(row['header']) if pd.notna(row['header']) else ""
        route_spans = parse_json_field(row.get('affected_spans', '[]'))
        direction_spans = parse_json_field(row.get('direction_spans', '[]'))
        
        aligned = align_spans_to_tokens(
            text, route_spans, direction_spans, tokenizer
        )
        
        all_input_ids.append(aligned["input_ids"])
        all_attention_masks.append(aligned["attention_mask"])
        all_labels.append(aligned["labels"])
    
    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })
    
    print(f"  Created NER dataset with {len(dataset):,} samples")
    return dataset


# ============================================================================
# RE: ENTITY MARKER INSERTION + RELATION LABELS
# ============================================================================

def insert_entity_markers(
    text: str,
    route_spans: List[Dict],
    direction_spans: List[Dict],
    route_id: int,
    direction_id: int
) -> str:
    """
    Insert entity markers around the specified route and direction spans.
    
    Example: "Southbound Q trains" -> "[DIR] Southbound [/DIR] [ROUTE] Q [/ROUTE] trains"
    """
    # Find the specific spans by ID
    route_span = next((s for s in route_spans if s['id'] == route_id), None)
    direction_span = next((s for s in direction_spans if s['id'] == direction_id), None)
    
    if not route_span or not direction_span:
        return text
    
    # Collect all insertions (position, marker, is_end)
    insertions = []
    
    # Route markers
    insertions.append((route_span['start'], "[ROUTE]", False))
    insertions.append((route_span['end'], "[/ROUTE]", True))
    
    # Direction markers
    insertions.append((direction_span['start'], "[DIR]", False))
    insertions.append((direction_span['end'], "[/DIR]", True))
    
    # Sort by position (descending) to insert from end to start
    # For same position: end markers before start markers
    insertions.sort(key=lambda x: (-x[0], x[2]))
    
    # Insert markers
    result = text
    for pos, marker, _ in insertions:
        result = result[:pos] + marker + " " + result[pos:]
    
    return result.strip()


def create_re_candidate_pairs(
    text: str,
    route_spans: List[Dict],
    direction_spans: List[Dict],
    relations: List[Dict],
    tokenizer,
    max_length: int = MAX_LENGTH
) -> List[Dict]:
    """
    Create candidate pairs for relation extraction.
    
    For each (route, direction) pair:
    - Insert entity markers
    - Label as 1 if HAS_DIRECTION relation exists, 0 otherwise
    
    Returns list of samples, each with:
    - input_ids, attention_mask (tokenized text with markers)
    - label (1 for HAS_DIRECTION, 0 for NO_RELATION)
    - route_id, direction_id (for reference)
    """
    samples = []
    
    if not route_spans or not direction_spans:
        return samples
    
    # Build set of positive relations for quick lookup
    positive_pairs = set()
    for rel in relations:
        positive_pairs.add((rel['route_span_id'], rel['direction_span_id']))
    
    # Create all candidate pairs
    for route in route_spans:
        for direction in direction_spans:
            route_id = route['id']
            direction_id = direction['id']
            
            # Insert entity markers
            marked_text = insert_entity_markers(
                text, route_spans, direction_spans, route_id, direction_id
            )
            
            # Tokenize
            encoding = tokenizer(
                marked_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None
            )
            
            # Determine label
            label = 1 if (route_id, direction_id) in positive_pairs else 0
            
            samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "label": label,
                "route_id": route_id,
                "direction_id": direction_id,
            })
    
    return samples


def prepare_re_dataset(df: pd.DataFrame, tokenizer, negative_ratio: float = 1.0) -> Dataset:
    """
    Prepare the RE dataset with entity markers and relation labels.
    
    Args:
        df: DataFrame with spans and relations
        tokenizer: HuggingFace tokenizer
        negative_ratio: Ratio of negative samples to keep (1.0 = keep all)
    """
    print("\nPreparing RE dataset...")
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_route_ids = []
    all_direction_ids = []
    
    positive_count = 0
    negative_count = 0
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {total:,} records...")
        
        text = str(row['header']) if pd.notna(row['header']) else ""
        route_spans = parse_json_field(row.get('affected_spans', '[]'))
        direction_spans = parse_json_field(row.get('direction_spans', '[]'))
        relations = parse_json_field(row.get('relations', '[]'))
        
        samples = create_re_candidate_pairs(
            text, route_spans, direction_spans, relations, tokenizer
        )
        
        for sample in samples:
            # Optionally downsample negatives
            if sample["label"] == 0:
                negative_count += 1
                if negative_ratio < 1.0:
                    import random
                    if random.random() > negative_ratio:
                        continue
            else:
                positive_count += 1
            
            all_input_ids.append(sample["input_ids"])
            all_attention_masks.append(sample["attention_mask"])
            all_labels.append(sample["label"])
            all_route_ids.append(sample["route_id"])
            all_direction_ids.append(sample["direction_id"])
    
    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
        "route_id": all_route_ids,
        "direction_id": all_direction_ids,
    })
    
    print(f"  Created RE dataset with {len(dataset):,} samples")
    print(f"  Positive pairs: {positive_count:,}")
    print(f"  Negative pairs: {negative_count:,}")
    
    return dataset


# ============================================================================
# DATASET SPLITTING
# ============================================================================

def split_dataset(dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15) -> DatasetDict:
    """
    Split dataset into train/val/test sets.
    """
    # First split: train vs (val + test)
    train_test = dataset.train_test_split(test_size=(1 - train_ratio), seed=42)
    
    # Second split: val vs test (from the remaining data)
    val_test_ratio = val_ratio / (1 - train_ratio)
    val_test = train_test["test"].train_test_split(test_size=(1 - val_test_ratio), seed=42)
    
    return DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for data preparation."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer (use slow tokenizer to avoid conversion issues)
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens for RE entity markers
    special_tokens = ["[ROUTE]", "[/ROUTE]", "[DIR]", "[/DIR]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added special tokens: {special_tokens}")
    print(f"Vocab size: {len(tokenizer):,}")
    
    # Save tokenizer for later use
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Load data
    df = load_data(INPUT_FILE)
    
    # ========== NER DATASET ==========
    ner_dataset = prepare_ner_dataset(df, tokenizer)
    ner_splits = split_dataset(ner_dataset)
    
    ner_path = os.path.join(OUTPUT_DIR, "ner_dataset")
    ner_splits.save_to_disk(ner_path)
    print(f"\nSaved NER dataset to {ner_path}")
    print(f"  Train: {len(ner_splits['train']):,}")
    print(f"  Validation: {len(ner_splits['validation']):,}")
    print(f"  Test: {len(ner_splits['test']):,}")
    
    # ========== RE DATASET ==========
    re_dataset = prepare_re_dataset(df, tokenizer, negative_ratio=1.0)
    re_splits = split_dataset(re_dataset)
    
    re_path = os.path.join(OUTPUT_DIR, "re_dataset")
    re_splits.save_to_disk(re_path)
    print(f"\nSaved RE dataset to {re_path}")
    print(f"  Train: {len(re_splits['train']):,}")
    print(f"  Validation: {len(re_splits['validation']):,}")
    print(f"  Test: {len(re_splits['test']):,}")
    
    # ========== SAVE LABEL MAPPINGS ==========
    label_info = {
        "ner_label2id": LABEL2ID,
        "ner_id2label": ID2LABEL,
        "re_label2id": {"NO_RELATION": 0, "HAS_DIRECTION": 1},
        "re_id2label": {0: "NO_RELATION", 1: "HAS_DIRECTION"},
    }
    
    label_path = os.path.join(OUTPUT_DIR, "label_mappings.json")
    with open(label_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"\nSaved label mappings to {label_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("  ├── tokenizer/          (DeBERTa tokenizer with special tokens)")
    print("  ├── ner_dataset/        (Token-level BIO labels)")
    print("  ├── re_dataset/         (Entity markers + relation labels)")
    print("  └── label_mappings.json (Label ID mappings)")
    print("\nNER Labels:")
    for label, idx in LABEL2ID.items():
        print(f"  {idx}: {label}")
    print(f"  {IGNORE_INDEX}: [IGNORED] (special tokens)")
    print("=" * 60)


if __name__ == "__main__":
    main()
