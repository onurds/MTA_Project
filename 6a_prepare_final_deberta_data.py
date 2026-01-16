"""
Code for Preparing Final Data for NER and RE Training

This script converts character level span annotations to:
1. NER: Token level BIO labels aligned with DeBERTa tokenizer
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

# CONFIGURATION

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
DEFAULT_INPUT_FILE = "Preprocessed/MTA_Data_silver_relations.csv"
DEFAULT_OUTPUT_DIR = "final_data"

# Dataset variants to materialize. Silver mirrors training data, gold is for evaluation
DATA_VARIANTS = {
    "silver": {
        "input_file": DEFAULT_INPUT_FILE,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "route_spans_col": "affected_spans",
        "direction_spans_col": "direction_spans",
        "relations_col": "relations",
        "split_strategy": "temporal",  # keep temporal split for training
    },
    "gold": {
        "input_file": "Preprocessed/MTA_Data_Final_Gold.csv",
        "output_dir": "final_data_gold",
        "route_spans_col": "affected_spans_gold",
        "direction_spans_col": "direction_spans_gold",
        "relations_col": "relations_gold",
        # Put all gold samples into the test split; no train/val for gold
        "split_strategy": "all_test",
    },
}

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



# DATA LOADING

def load_data(filepath: str) -> pd.DataFrame:
    # Load the silver relations dataset.
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df


def parse_json_field(json_str: str) -> List[Dict]:
    # Parse JSON string field to list of dicts.
    if pd.isna(json_str) or json_str == '[]' or json_str == '':
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def filter_valid_spans(spans: List[Dict], span_type: str) -> List[Dict]:
    # Drop malformed span dicts (missing start/end) to avoid crashing downstream.
    valid = []
    for s in spans:
        if isinstance(s, dict) and 'start' in s and 'end' in s:
            valid.append(s)
        else:
            # Skip bad spans silently; counts are small and logged per row later
            continue
    if len(valid) != len(spans):
        print(f"  Skipped {len(spans) - len(valid)} malformed {span_type} spans")
    return valid


# NER: CHARACTER TO TOKEN ALIGNMENT

def compute_token_offsets(text: str, tokenizer) -> List[Tuple[int, int]]:
    # Compute character offsets for each token manually.
    # needed because the slow tokenizer doesn't support return_offsets_mapping
    
    tokens = tokenizer.tokenize(text)
    offsets = []
    current_pos = 0
    
    for token in tokens:
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
    
    # Convert character-level spans to token-level BIO labels.
    
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


def prepare_ner_dataset(
    df: pd.DataFrame,
    tokenizer,
    span_cols: Optional[Dict[str, str]] = None
) -> Dataset:
    
    # Prepare the full NER dataset with token aligned BIO labels.
    
    print("\nPreparing NER dataset...")
    span_cols = span_cols or {
        "route": "affected_spans",
        "direction": "direction_spans",
    }
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {total:,} records...")
        
        text = str(row['header']) if pd.notna(row['header']) else ""
        route_spans_raw = parse_json_field(row.get(span_cols["route"], '[]'))
        direction_spans_raw = parse_json_field(row.get(span_cols["direction"], '[]'))

        route_spans = filter_valid_spans(route_spans_raw, "route")
        direction_spans = filter_valid_spans(direction_spans_raw, "direction")
        
        try:
            aligned = align_spans_to_tokens(
                text, route_spans, direction_spans, tokenizer
            )
        except KeyError as e:
            # Skip row if spans are still malformed
            print(f"  Skipping row {idx} due to span KeyError: {e}")
            continue
        
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


# RE: ENTITY MARKER INSERTION + RELATION LABELS

def insert_entity_markers(
    text: str,
    route_spans: List[Dict],
    direction_spans: List[Dict],
    route_id: int,
    direction_id: int
) -> str:
    # Insert entity markers around the specified route and direction spans.
    
    # Example: "Southbound Q trains" - "[DIR] Southbound [/DIR] [ROUTE] Q [/ROUTE] trains"
    
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


def prepare_re_dataset(
    df: pd.DataFrame,
    tokenizer,
    negative_ratio: float = 1.0,
    span_cols: Optional[Dict[str, str]] = None
) -> Dataset:
    # Prepare the RE dataset with entity markers and relation labels.
    
    print("\nPreparing RE dataset...")
    span_cols = span_cols or {
        "route": "affected_spans",
        "direction": "direction_spans",
        "relations": "relations",
    }
    
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
        route_spans_raw = parse_json_field(row.get(span_cols["route"], '[]'))
        direction_spans_raw = parse_json_field(row.get(span_cols["direction"], '[]'))
        relations = parse_json_field(row.get(span_cols["relations"], '[]'))

        route_spans = filter_valid_spans(route_spans_raw, "route")
        direction_spans = filter_valid_spans(direction_spans_raw, "direction")
        
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


# DATASET SPLITTING

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    strategy: str = "temporal"
) -> DatasetDict:
    
    # Split dataset into train/val/test sets.
    
    n = len(dataset)
    
    if strategy == "all_test":
        print("\n  Using all samples in test split (no train/val) for gold evaluation")
        empty = dataset.select([])  # empty dataset
        dataset_dict = DatasetDict({
            "train": empty,
            "validation": empty,
            "test": dataset,
        })
        print(f"    Train: {len(dataset_dict['train']):,} samples")
        print(f"    Val:   {len(dataset_dict['validation']):,} samples")
        print(f"    Test:  {len(dataset_dict['test']):,} samples (all gold)")
        return dataset_dict
    
    # Default: temporal split
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Use select() to slice the dataset
    train_dataset = dataset.select(range(0, train_end))
    val_dataset = dataset.select(range(train_end, val_end))
    test_dataset = dataset.select(range(val_end, n))
    
    print(f"\n  Temporal split (no shuffling):")
    print(f"    Train: indices 0-{train_end-1} ({len(train_dataset):,} samples)")
    print(f"    Val:   indices {train_end}-{val_end-1} ({len(val_dataset):,} samples)")
    print(f"    Test:  indices {val_end}-{n-1} ({len(test_dataset):,} samples)")
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })


# MAIN FUNCTION

def main():
    
    # Load tokenizer (use slow tokenizer to avoid conversion issues)
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens for RE entity markers
    special_tokens = ["[ROUTE]", "[/ROUTE]", "[DIR]", "[/DIR]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added special tokens: {special_tokens}")
    print(f"Vocab size: {len(tokenizer):,}")
    
    # Save tokenizer for later use (shared by all variants)
    tokenizer_path = os.path.join(DEFAULT_OUTPUT_DIR, "tokenizer")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Process each dataset variant (silver for training, gold for evaluation)
    for variant_name, cfg in DATA_VARIANTS.items():
        print("\n" + "=" * 60)
        print(f"PROCESSING {variant_name.upper()} DATA")
        print("=" * 60)
        
        output_dir = cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = load_data(cfg["input_file"])
        
        # IMPORTANT: Sort by date for temporal splitting
        print("\nSorting by date for temporal split...")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        span_cols = {
            "route": cfg["route_spans_col"],
            "direction": cfg["direction_spans_col"],
            "relations": cfg["relations_col"],
        }
        split_strategy = cfg.get("split_strategy", "temporal")
        
        # ========== NER DATASET ==========
        ner_dataset = prepare_ner_dataset(df, tokenizer, span_cols=span_cols)
        ner_splits = split_dataset(ner_dataset, strategy=split_strategy)
        
        ner_path = os.path.join(output_dir, "ner_dataset")
        ner_splits.save_to_disk(ner_path)
        print(f"\nSaved NER dataset to {ner_path}")
        print(f"  Train: {len(ner_splits['train']):,}")
        print(f"  Validation: {len(ner_splits['validation']):,}")
        print(f"  Test: {len(ner_splits['test']):,}")
        
        # ========== RE DATASET ==========
        re_dataset = prepare_re_dataset(df, tokenizer, negative_ratio=1.0, span_cols=span_cols)
        re_splits = split_dataset(re_dataset, strategy=split_strategy)
        
        re_path = os.path.join(output_dir, "re_dataset")
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
        
        label_path = os.path.join(output_dir, "label_mappings.json")
        with open(label_path, "w") as f:
            json.dump(label_info, f, indent=2)
        print(f"\nSaved label mappings to {label_path}")
        
        # ========== SUMMARY ==========
        print(f"\n{variant_name.upper()} data preparation complete.")
        print(f"Output directory: {output_dir}/")
        print("  tokenizer/          (shared from final_data/tokenizer)")
        print("  ner_dataset/        (Token-level BIO labels)")
        print("  re_dataset/         (Entity markers + relation labels)")
        print("  label_mappings.json (Label ID mappings)")
        print("NER Labels:")
        for label, idx in LABEL2ID.items():
            print(f"  {idx}: {label}")
        print(f"  {IGNORE_INDEX}: [IGNORED] (special tokens)")


if __name__ == "__main__":
    main()
