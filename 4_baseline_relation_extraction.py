"""
Baseline Relation Extraction for MTA Transit Alerts

This module implements a rule-based relation extraction system that pairs
DIRECTION and ROUTE entities using segment-based logic with direction inheritance.

Schema (paper-ready):
- affected_spans: [{"id": 0, "start": X, "end": Y, "type": "ROUTE", "value": "Q"}, ...]
- direction_spans: [{"id": 0, "start": X, "end": Y, "type": "DIRECTION", "value": "SOUTHBOUND"}, ...]
- relations: [{"route_span_id": 0, "direction_span_id": 0, "type": "HAS_DIRECTION"}, ...]

Rules:
1. Direction inheritance: Active direction persists until a new direction is found
2. Major segment breaks (reset active direction): newlines, parentheses, colons (not followed by time)
3. "both directions" post-route: Updates preceding route's pairing
4. Sequential left-to-right processing of merged entity spans
"""

import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def load_silver_data(filepath: str) -> pd.DataFrame:
    """Load the silver dataset with direction and route spans."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df


def parse_spans(spans_json: str) -> List[Dict]:
    """Parse JSON spans string into list of span dictionaries."""
    if pd.isna(spans_json) or spans_json == '[]':
        return []
    try:
        return json.loads(spans_json)
    except (json.JSONDecodeError, TypeError):
        return []


def add_span_ids(spans: List[Dict], start_id: int = 0) -> List[Dict]:
    """Add sequential IDs to spans if not already present."""
    result = []
    for i, span in enumerate(spans):
        span_with_id = span.copy()
        if 'id' not in span_with_id:
            span_with_id['id'] = start_id + i
        result.append(span_with_id)
    return result


def is_major_break(text_between: str) -> bool:
    """
    Check if text between two entities contains a major segment break.
    Major breaks: newlines, parentheses, colons (not followed by time pattern)
    """
    # Check for newline
    if '\n' in text_between:
        return True
    
    # Check for parentheses
    if '(' in text_between or ')' in text_between:
        return True
    
    # Check for colon not followed by time pattern
    # Time pattern: colon followed by 2 digits (e.g., "8:45", "10:30")
    colon_matches = list(re.finditer(r':', text_between))
    for match in colon_matches:
        after_colon = text_between[match.end():]
        # If colon is NOT followed by time pattern (digits), it's a major break
        if not re.match(r'\s*\d{2}', after_colon):
            return True
    
    return False


def extract_relations(
    header: str, 
    direction_spans: List[Dict], 
    route_spans: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extract HAS_DIRECTION relations from a header using segment-based
    pairing with direction inheritance.
    
    Args:
        header: The alert header text
        direction_spans: List of direction span dicts (with IDs)
        route_spans: List of route span dicts (with IDs)
    
    Returns:
        Tuple of (updated_route_spans, updated_direction_spans, relations)
        where relations use span IDs: {"route_span_id": X, "direction_span_id": Y, "type": "HAS_DIRECTION"}
    
    Rules:
    - Iterate left-to-right through merged entities
    - When DIRECTION found: set as active direction
    - When ROUTE found: pair with active direction (if any)
    - Reset active direction on major breaks (newline, parentheses, non-time colon)
    - Handle "both directions" post-route pattern (direction comes after route)
    - Second pass: assign unpaired routes to next available direction in same segment
    """
    # Add IDs to spans
    route_spans_with_ids = add_span_ids(route_spans, start_id=0)
    direction_spans_with_ids = add_span_ids(direction_spans, start_id=0)
    
    if not direction_spans_with_ids or not route_spans_with_ids:
        return route_spans_with_ids, direction_spans_with_ids, []
    
    # Create lookup maps for quick ID access
    route_id_map = {(s['start'], s['end']): s['id'] for s in route_spans_with_ids}
    direction_id_map = {(s['start'], s['end']): s['id'] for s in direction_spans_with_ids}
    
    # Merge all entities for sequential processing
    entities = []
    for span in direction_spans_with_ids:
        entities.append({
            'start': span['start'],
            'end': span['end'],
            'id': span['id'],
            'value': span['value'],
            'entity_type': 'DIRECTION'
        })
    for span in route_spans_with_ids:
        entities.append({
            'start': span['start'],
            'end': span['end'],
            'id': span['id'],
            'value': span['value'],
            'entity_type': 'ROUTE'
        })
    
    # Sort by start position
    entities.sort(key=lambda x: x['start'])
    
    relations = []
    active_direction_id = None
    active_direction_end = None
    
    # Track route_id -> direction_id pairings
    route_direction_pairs = {}
    
    # Track unpaired routes for second pass
    unpaired_routes = []
    
    for i, entity in enumerate(entities):
        # Check for major break from previous entity
        if i > 0:
            prev_entity = entities[i - 1]
            text_between = header[prev_entity['end']:entity['start']]
            
            if is_major_break(text_between):
                active_direction_id = None
                active_direction_end = None
        
        if entity['entity_type'] == 'DIRECTION':
            # Update active direction
            active_direction_id = entity['id']
            active_direction_end = entity['end']
        
        elif entity['entity_type'] == 'ROUTE':
            route_id = entity['id']
            
            if active_direction_id is not None:
                route_direction_pairs[route_id] = active_direction_id
            else:
                # Track unpaired route for second pass
                unpaired_routes.append({
                    'route_id': route_id,
                    'route_start': entity['start'],
                    'route_end': entity['end']
                })
    
    # Second pass: assign unpaired routes to the next direction in the same segment
    # This handles "L trains are delayed in both directions" pattern
    for unpaired in unpaired_routes:
        route_id = unpaired['route_id']
        route_end = unpaired['route_end']
        
        # Find the next direction after this route
        for entity in entities:
            if entity['entity_type'] == 'DIRECTION' and entity['start'] > route_end:
                # Check if there's a segment break between route and this direction
                text_between = header[route_end:entity['start']]
                if not is_major_break(text_between):
                    route_direction_pairs[route_id] = entity['id']
                    break
                else:
                    # Hit a segment break, stop looking
                    break
    
    # Convert pairs to relation format
    for route_id, direction_id in route_direction_pairs.items():
        relations.append({
            'route_span_id': route_id,
            'direction_span_id': direction_id,
            'type': 'HAS_DIRECTION'
        })
    
    # Sort relations by route_span_id for consistent output
    relations.sort(key=lambda x: x['route_span_id'])
    
    return route_spans_with_ids, direction_spans_with_ids, relations


def process_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Process the silver dataset and extract relations for each row.
    Updates spans to include IDs and adds relations column.
    
    Args:
        input_path: Path to silver dataset CSV
        output_path: Path to output CSV with relations
    
    Returns:
        DataFrame with updated spans and relations column
    """
    df = load_silver_data(input_path)
    
    print("Extracting relations...")
    updated_route_spans_list = []
    updated_direction_spans_list = []
    relations_list = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,} records...")
        
        header = str(row['header']) if pd.notna(row['header']) else ""
        direction_spans = parse_spans(row.get('direction_spans', '[]'))
        route_spans = parse_spans(row.get('affected_spans', '[]'))
        
        updated_routes, updated_directions, relations = extract_relations(
            header, direction_spans, route_spans
        )
        
        updated_route_spans_list.append(json.dumps(updated_routes))
        updated_direction_spans_list.append(json.dumps(updated_directions))
        relations_list.append(json.dumps(relations))
    
    # Update dataframe with ID-enhanced spans and relations
    df['affected_spans'] = updated_route_spans_list
    df['direction_spans'] = updated_direction_spans_list
    df['relations'] = relations_list
    
    print(f"\nWriting output to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote {len(df):,} records to {output_path}")
    
    return df


def print_eda_stats(df: pd.DataFrame):
    """Print compact EDA statistics for the extracted relations."""
    print("\n" + "=" * 60)
    print("RELATION EXTRACTION STATISTICS")
    print("=" * 60)
    
    # Parse relations and spans for analysis
    relations_counts = []
    direction_type_counts = defaultdict(int)
    total_relations = 0
    rows_with_relations = 0
    
    for idx, row in df.iterrows():
        relations = json.loads(row['relations']) if row['relations'] else []
        direction_spans = json.loads(row['direction_spans']) if row['direction_spans'] else []
        
        # Build direction_id -> value map
        dir_id_to_value = {s['id']: s['value'] for s in direction_spans}
        
        count = len(relations)
        relations_counts.append(count)
        total_relations += count
        
        if count > 0:
            rows_with_relations += 1
        
        for rel in relations:
            direction_id = rel['direction_span_id']
            direction_value = dir_id_to_value.get(direction_id, 'UNKNOWN')
            direction_type_counts[direction_value] += 1
    
    # Basic stats
    print(f"\nTotal rows: {len(df):,}")
    print(f"Rows with relations: {rows_with_relations:,} ({100*rows_with_relations/len(df):.1f}%)")
    print(f"Total relation pairs: {total_relations:,}")
    
    # Single vs multi-relation distribution
    single_relation = sum(1 for c in relations_counts if c == 1)
    multi_relation = sum(1 for c in relations_counts if c > 1)
    print(f"\nSingle-relation rows: {single_relation:,}")
    print(f"Multi-relation rows: {multi_relation:,}")
    
    # Direction type distribution
    print(f"\nDirection Type Distribution:")
    for direction, count in sorted(direction_type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_relations if total_relations > 0 else 0
        print(f"  {direction}: {count:,} ({pct:.1f}%)")
    
    print("=" * 60)


def main():
    """Main entry point for baseline relation extraction."""
    input_path = 'Preprocessed/MTA_Data_silver_directions.csv'
    output_path = 'Preprocessed/MTA_Data_silver_relations.csv'
    
    df = process_dataset(input_path, output_path)
    print_eda_stats(df)


if __name__ == "__main__":
    main()
