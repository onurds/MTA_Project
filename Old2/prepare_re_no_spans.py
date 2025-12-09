"""
Prepare Relation Extraction Data Without Spans (Option 2)

This script converts span-based relation annotations to a span-free format
suitable for BiLSTM/DeBERTa relation extraction models.

Input: MTA_Data_silver_relations.csv (with affected_spans, direction_spans, relations)
Output: MTA_Data_RE_no_spans.csv (entity-marker format for binary classification)

Each (route, direction) candidate pair becomes one example:
- Positive (label=1): The pair exists in the original relations
- Negative (label=0): The pair is a candidate but not a true relation

Key Features:
- Extracts actual direction TEXT from spans (handles PLACE_BOUND like "Jamaica-bound")
- Properly handles multiple routes and multiple directions per alert
- Creates all candidate pairs for relation classification
"""

import pandas as pd
import json
import re
from typing import List, Dict, Set, Tuple
from itertools import product
from collections import Counter


def parse_json_field(value) -> List:
    """Safely parse a JSON string field."""
    if pd.isna(value) or value == '[]' or value == '':
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def extract_entities_and_relations(row) -> Dict:
    """
    Extract entity values (with actual text for directions) and convert span-based relations to value-based.
    
    For PLACE_BOUND directions, extracts the actual text (e.g., "Jamaica-bound") from the header
    using the span's start/end positions.
    
    Returns:
        {
            'routes': [{'id': 0, 'value': 'Q', 'text': 'Q'}, ...],
            'directions': [{'id': 0, 'value': 'NORTHBOUND', 'text': 'Northbound'}, ...],
            'relations': [(route_id, direction_id), ...]
        }
    """
    header = str(row.get('header', '')) if pd.notna(row.get('header')) else ''
    affected_spans = parse_json_field(row.get('affected_spans', '[]'))
    direction_spans = parse_json_field(row.get('direction_spans', '[]'))
    relations = parse_json_field(row.get('relations', '[]'))
    
    # Build route info (ID -> {value, text})
    routes = []
    for span in affected_spans:
        if 'id' not in span or 'value' not in span:
            continue
        
        # Extract actual text from header using span positions
        start = span.get('start', 0)
        end = span.get('end', 0)
        text = header[start:end] if start < end <= len(header) else span['value']
        
        routes.append({
            'id': span['id'],
            'value': span['value'],
            'text': text
        })
    
    # Build direction info (ID -> {value, text})
    directions = []
    for span in direction_spans:
        if 'id' not in span or 'value' not in span:
            continue
        
        # Extract actual text from header using span positions
        start = span.get('start', 0)
        end = span.get('end', 0)
        text = header[start:end] if start < end <= len(header) else span['value']
        
        directions.append({
            'id': span['id'],
            'value': span['value'],
            'text': text  # This captures "Jamaica-bound", "8 Av-bound", etc.
        })
    
    # Convert relations from ID-based to (route_id, direction_id) tuples
    relation_pairs = set()
    for rel in relations:
        route_id = rel.get('route_span_id')
        dir_id = rel.get('direction_span_id')
        if route_id is not None and dir_id is not None:
            relation_pairs.add((route_id, dir_id))
    
    return {
        'routes': routes,
        'directions': directions,
        'relations': relation_pairs
    }


def insert_entity_markers(text: str, route_text: str, direction_text: str) -> str:
    """
    Insert entity markers around the ACTUAL text of each entity in the text.
    
    Markers:
        - [R] route [/R] for routes
        - [D] direction [/D] for directions
    
    Uses the actual extracted text (not just the label) so it matches correctly,
    especially for PLACE_BOUND directions like "Jamaica-bound".
    
    Returns the marked text for the given (route, direction) pair.
    """
    if not text:
        return ""
    
    marked = text
    route_marked = False
    dir_marked = False
    
    # Find and mark the route (exact text match)
    route_pattern = re.compile(re.escape(route_text), re.IGNORECASE)
    route_match = route_pattern.search(marked)
    
    if route_match:
        start, end = route_match.start(), route_match.end()
        matched_text = route_match.group()
        marked = marked[:start] + f"[R] {matched_text} [/R]" + marked[end:]
        route_marked = True
    
    # Find and mark the direction (exact text match)
    # The direction_text is the actual text like "Jamaica-bound", "Northbound", etc.
    dir_pattern = re.compile(re.escape(direction_text), re.IGNORECASE)
    dir_match = dir_pattern.search(marked)
    
    if dir_match:
        start, end = dir_match.start(), dir_match.end()
        matched_text = dir_match.group()
        # Make sure we're not matching inside the route markers
        if route_marked:
            r_start = marked.find('[R]')
            r_end = marked.find('[/R]') + 4
            if start >= r_start and end <= r_end:
                # Try to find another occurrence
                dir_match = dir_pattern.search(marked, pos=r_end)
                if dir_match:
                    start, end = dir_match.start(), dir_match.end()
                    matched_text = dir_match.group()
                else:
                    return marked
        marked = marked[:start] + f"[D] {matched_text} [/D]" + marked[end:]
        dir_marked = True
    
    return marked


def create_re_examples(df: pd.DataFrame, include_negatives: bool = True) -> List[Dict]:
    """
    Create relation extraction training examples.
    
    For each row, generates examples for all candidate (route, direction) pairs.
    - Positive examples: pairs that exist in the original relations
    - Negative examples: pairs that are candidates but not true relations
    
    Args:
        df: DataFrame with columns header, affected_spans, direction_spans, relations
        include_negatives: Whether to include negative examples (default True)
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processing row {idx:,}...")
        
        header = str(row['header']) if pd.notna(row.get('header')) else ""
        
        if not header.strip():
            continue
        
        # Extract entities and relations
        entity_data = extract_entities_and_relations(row)
        routes = entity_data['routes']
        directions = entity_data['directions']
        positive_pairs = entity_data['relations']  # Set of (route_id, dir_id)
        
        # Skip rows with no routes or directions
        if not routes or not directions:
            continue
        
        # Generate examples for all candidate pairs
        for route_info in routes:
            for dir_info in directions:
                route_id = route_info['id']
                dir_id = dir_info['id']
                
                is_positive = (route_id, dir_id) in positive_pairs
                
                # Optionally skip negatives
                if not include_negatives and not is_positive:
                    continue
                
                # Create marked text using ACTUAL text (not just labels)
                marked_text = insert_entity_markers(
                    header, 
                    route_info['text'],  # Actual route text
                    dir_info['text']     # Actual direction text (e.g., "Jamaica-bound")
                )
                
                examples.append({
                    'original_idx': idx,
                    'original_text': header,
                    'marked_text': marked_text,
                    'route': route_info['value'],           # Route code (e.g., "Q", "J")
                    'route_text': route_info['text'],       # Actual text in header
                    'direction': dir_info['value'],         # Direction label (e.g., "PLACE_BOUND")
                    'direction_text': dir_info['text'],     # Actual text (e.g., "Jamaica-bound")
                    'label': 1 if is_positive else 0
                })
    
    return examples


def create_multi_label_examples(df: pd.DataFrame) -> List[Dict]:
    """
    Alternative format: One row per text with ALL relations as a list.
    
    This format is better for models that can predict multiple relations at once.
    
    Returns:
        List of dictionaries with 'text', 'routes', 'directions', 'relations'
    """
    examples = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processing row {idx:,}...")
        
        header = str(row['header']) if pd.notna(row.get('header')) else ""
        
        if not header.strip():
            continue
        
        entity_data = extract_entities_and_relations(row)
        routes = entity_data['routes']
        directions = entity_data['directions']
        positive_pairs = entity_data['relations']
        
        if not routes or not directions:
            continue
        
        # Build value-based relations
        route_id_to_info = {r['id']: r for r in routes}
        dir_id_to_info = {d['id']: d for d in directions}
        
        relations = []
        for route_id, dir_id in positive_pairs:
            route_info = route_id_to_info.get(route_id)
            dir_info = dir_id_to_info.get(dir_id)
            if route_info and dir_info:
                relations.append({
                    'route': route_info['value'],
                    'route_text': route_info['text'],
                    'direction': dir_info['value'],
                    'direction_text': dir_info['text']
                })
        
        examples.append({
            'original_idx': idx,
            'text': header,
            'routes': json.dumps([r['value'] for r in routes]),
            'route_texts': json.dumps([r['text'] for r in routes]),
            'directions': json.dumps([d['value'] for d in directions]),
            'direction_texts': json.dumps([d['text'] for d in directions]),
            'relations': json.dumps(relations),
            'num_relations': len(relations)
        })
    
    return examples


def print_statistics(df: pd.DataFrame, examples_df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nOriginal Data:")
    print(f"  Total rows: {len(df):,}")
    
    print(f"\nRE Examples (No Spans - Pair Classification):")
    print(f"  Total examples: {len(examples_df):,}")
    print(f"  Positive (HAS_DIRECTION): {examples_df['label'].sum():,} ({examples_df['label'].mean()*100:.1f}%)")
    print(f"  Negative (NO_RELATION): {(examples_df['label'] == 0).sum():,} ({(1-examples_df['label'].mean())*100:.1f}%)")
    
    print(f"\nUnique Values:")
    print(f"  Unique routes: {examples_df['route'].nunique()}")
    print(f"  Unique direction labels: {examples_df['direction'].nunique()}")
    print(f"  Unique direction texts: {examples_df['direction_text'].nunique()}")
    
    print(f"\nDirection Label Distribution (positive examples):")
    dir_counts = examples_df[examples_df['label'] == 1]['direction'].value_counts()
    for direction, count in dir_counts.head(15).items():
        print(f"    {direction}: {count:,}")
    
    print("\n" + "=" * 60)


def show_examples(examples_df: pd.DataFrame, n: int = 3):
    """Display sample examples."""
    print("\nSAMPLE EXAMPLES")
    print("-" * 60)
    
    # Show positive examples
    positives = examples_df[examples_df['label'] == 1].head(n)
    print("\nPositive Examples (label=1):")
    for _, row in positives.iterrows():
        print(f"\n  Original: {row['original_text'][:100]}...")
        print(f"  Marked:   {row['marked_text'][:100]}...")
        print(f"  Route: {row['route']} (text: '{row['route_text']}')")
        print(f"  Direction: {row['direction']} (text: '{row['direction_text']}')")
    
    # Show PLACE_BOUND examples specifically
    place_bounds = examples_df[(examples_df['label'] == 1) & (examples_df['direction'] == 'PLACE_BOUND')].head(n)
    if len(place_bounds) > 0:
        print("\n\nPLACE_BOUND Examples (label=1):")
        for _, row in place_bounds.iterrows():
            print(f"\n  Original: {row['original_text'][:100]}...")
            print(f"  Marked:   {row['marked_text'][:100]}...")
            print(f"  Route: {row['route']} (text: '{row['route_text']}')")
            print(f"  Direction: {row['direction']} (text: '{row['direction_text']}')")
    
    # Show negative examples
    negatives = examples_df[examples_df['label'] == 0].head(n)
    if len(negatives) > 0:
        print("\n\nNegative Examples (label=0):")
        for _, row in negatives.iterrows():
            print(f"\n  Original: {row['original_text'][:100]}...")
            print(f"  Marked:   {row['marked_text'][:100]}...")
            print(f"  Route: {row['route']} (text: '{row['route_text']}')")
            print(f"  Direction: {row['direction']} (text: '{row['direction_text']}')")


def main():
    # Configuration
    input_path = 'Preprocessed/MTA_Data_silver_relations.csv'
    output_path = 'Preprocessed/MTA_Data_RE_no_spans.csv'
    output_multilabel_path = 'Preprocessed/MTA_Data_RE_multilabel.csv'
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    
    # ====================
    # Option A: Pair Classification Format
    # Each (route, direction) candidate = one row with binary label
    # ====================
    print("\n" + "=" * 60)
    print("Creating PAIR CLASSIFICATION examples...")
    print("(Each route-direction candidate pair = one training example)")
    print("=" * 60)
    
    examples = create_re_examples(df, include_negatives=True)
    examples_df = pd.DataFrame(examples)
    
    print_statistics(df, examples_df)
    show_examples(examples_df)
    
    # Save pair classification format
    examples_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(examples_df):,} examples to {output_path}")
    
    # Save training-ready version (minimal columns)
    training_cols = ['marked_text', 'route', 'direction', 'direction_text', 'label']
    training_df = examples_df[training_cols]
    training_path = output_path.replace('.csv', '_training.csv')
    training_df.to_csv(training_path, index=False)
    print(f"Saved training-ready version to {training_path}")
    
    # ====================
    # Option B: Multi-Label Format
    # One row per text with all relations as JSON list
    # ====================
    print("\n" + "=" * 60)
    print("Creating MULTI-LABEL examples...")
    print("(One row per text with all relations as a list)")
    print("=" * 60)
    
    multilabel_examples = create_multi_label_examples(df)
    multilabel_df = pd.DataFrame(multilabel_examples)
    
    print(f"\nMulti-Label Format Statistics:")
    print(f"  Total rows: {len(multilabel_df):,}")
    print(f"  Rows with 1 relation: {(multilabel_df['num_relations'] == 1).sum():,}")
    print(f"  Rows with 2+ relations: {(multilabel_df['num_relations'] >= 2).sum():,}")
    print(f"  Rows with 5+ relations: {(multilabel_df['num_relations'] >= 5).sum():,}")
    print(f"  Max relations in one row: {multilabel_df['num_relations'].max()}")
    
    multilabel_df.to_csv(output_multilabel_path, index=False)
    print(f"\nSaved {len(multilabel_df):,} rows to {output_multilabel_path}")
    
    # Show multi-label examples
    print("\nSample Multi-Label Rows:")
    for _, row in multilabel_df[multilabel_df['num_relations'] >= 2].head(3).iterrows():
        print(f"\n  Text: {row['text'][:80]}...")
        print(f"  Routes: {row['routes']}")
        print(f"  Directions: {row['direction_texts']}")
        print(f"  Relations: {row['relations']}")


if __name__ == "__main__":
    main()
