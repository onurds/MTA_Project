"""
Prepare Final Data for NER Training

This script converts character-level span annotations to token-level BIO labels 
aligned with DeBERTa tokenizer.

Input: Preprocessed/MTA_Data_silver_relations.csv
Output: 
  - final_data/ner_dataset (HuggingFace Dataset)
  - final_data/tokenizer (DeBERTa tokenizer)

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
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict
from transformers import DebertaV2Tokenizer
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128  # Reduced from 256 - MTA headers fit in <100 tokens
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
    # Tokenize WITHOUT padding - let DataCollator handle dynamic padding during training
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=False,  # Changed from "max_length" - massive speedup during training
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
    """Main entry point for NER data preparation."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer (use slow tokenizer to avoid conversion issues)
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
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
    
    # ========== SAVE LABEL MAPPINGS ==========
    label_info = {
        "ner_label2id": LABEL2ID,
        "ner_id2label": ID2LABEL,
    }
    
    label_path = os.path.join(OUTPUT_DIR, "label_mappings.json")
    with open(label_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"\nSaved label mappings to {label_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("NER DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("  ├── tokenizer/          (DeBERTa tokenizer)")
    print("  ├── ner_dataset/        (Token-level BIO labels)")
    print("  └── label_mappings.json (NER label ID mappings)")
    print("\nNER Labels:")
    for label, idx in LABEL2ID.items():
        print(f"  {idx}: {label}")
    print(f"  {IGNORE_INDEX}: [IGNORED] (special tokens)")
    print("=" * 60)


if __name__ == "__main__":
    main()
