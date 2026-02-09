"""
LLM Zero-Shot Evaluation Script V2
Compares LLM predictions against Gold annotations for transit alert extraction
Added: Export misclassified rows to CSV

Author: Onur Dursun
Dataset: MTA Service Alerts
"""

import pandas as pd
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

GOLD_CSV = "Preprocessed/MTA_Data_Final_Gold.csv"
LLM_CSV = "Preprocessed/MTA_Data_gold_dataset_LLM.csv"
OUTPUT_RESULTS = "Results/llm_evaluation_results.txt"
OUTPUT_MISCLASSIFIED = "Results/gold_LLM_misclassified_rows.csv"

# ============================================================================
# Helper Functions
# ============================================================================

def safe_json_loads(json_str):
    """Safely load JSON string, return empty list if invalid"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def extract_direction_type_stats(directions: List[Dict]) -> Dict[str, int]:
    """Count directions by type"""
    type_counts = defaultdict(int)
    for d in directions:
        if isinstance(d, dict) and 'type' in d:
            type_counts[d['type']] += 1
    return dict(type_counts)


# ============================================================================
# Main Evaluation Functions
# ============================================================================

def evaluate_ner(merged_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Evaluate Named Entity Recognition performance
    Returns metrics for routes and directions separately
    """
    metrics = {
        'routes': {'tp': 0, 'fp': 0, 'fn': 0},
        'directions': {'tp': 0, 'fp': 0, 'fn': 0},
        'direction_types': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    }
    
    for _, row in merged_df.iterrows():
        # Parse routes
        llm_routes = set(safe_json_loads(row['routes_llm']))
        gold_routes = set(safe_json_loads(row['routes_gold']))
        
        metrics['routes']['tp'] += len(llm_routes & gold_routes)
        metrics['routes']['fp'] += len(llm_routes - gold_routes)
        metrics['routes']['fn'] += len(gold_routes - llm_routes)
        
        # Parse directions
        llm_dirs_raw = safe_json_loads(row['directions_llm'])
        gold_dirs_raw = safe_json_loads(row['directions_gold'])
        
        llm_dirs = set(d['text'] if isinstance(d, dict) else d for d in llm_dirs_raw)
        gold_dirs = set(d['text'] if isinstance(d, dict) else d for d in gold_dirs_raw)
        
        metrics['directions']['tp'] += len(llm_dirs & gold_dirs)
        metrics['directions']['fp'] += len(llm_dirs - gold_dirs)
        metrics['directions']['fn'] += len(gold_dirs - llm_dirs)
        
        # Track by direction type
        llm_by_type = {}
        for d in llm_dirs_raw:
            if isinstance(d, dict) and 'text' in d and 'type' in d:
                llm_by_type[d['text']] = d['type']
        
        gold_by_type = {}
        for d in gold_dirs_raw:
            if isinstance(d, dict) and 'text' in d and 'type' in d:
                gold_by_type[d['text']] = d['type']
        
        # Calculate per-type metrics
        for dir_text in gold_dirs:
            if dir_text in gold_by_type:
                dir_type = gold_by_type[dir_text]
                if dir_text in llm_dirs:
                    metrics['direction_types'][dir_type]['tp'] += 1
                else:
                    metrics['direction_types'][dir_type]['fn'] += 1
        
        for dir_text in llm_dirs:
            if dir_text in llm_by_type and dir_text not in gold_dirs:
                dir_type = llm_by_type[dir_text]
                metrics['direction_types'][dir_type]['fp'] += 1
    
    # Calculate final metrics
    results = {
        'routes': calculate_metrics(**metrics['routes']),
        'directions': calculate_metrics(**metrics['directions']),
        'direction_types': {
            dtype: calculate_metrics(**counts) 
            for dtype, counts in metrics['direction_types'].items()
        }
    }
    
    return results


def evaluate_relations(merged_df: pd.DataFrame) -> Dict[str, any]:
    """
    Evaluate Relation Extraction performance
    Returns overall metrics and complexity-based breakdown
    """
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    by_complexity = {
        'simple': {'tp': 0, 'fp': 0, 'fn': 0},      # 1 route, 1 direction
        'moderate': {'tp': 0, 'fp': 0, 'fn': 0},    # 2-3 routes or directions
        'complex': {'tp': 0, 'fp': 0, 'fn': 0}      # 4+ routes or directions
    }
    
    for _, row in merged_df.iterrows():
        # Parse LLM relations (already has route and direction values)
        # Note: Converting to set automatically deduplicates repeated relations
        llm_rels_raw = safe_json_loads(row['relations_llm'])
        llm_rels = set(
            (r['route'], r['direction']) 
            for r in llm_rels_raw 
            if isinstance(r, dict) and 'route' in r and 'direction' in r
        )
        
        # Parse gold relations using span-based data
        # Note: Multiple span mentions may map to same relation; set deduplicates
        gold_rels_raw = safe_json_loads(row['relations_gold'])
        affected_spans = safe_json_loads(row['affected_spans_gold'])
        direction_spans = safe_json_loads(row['direction_spans_gold'])
        
        # Build mapping from span ID to value
        route_map = {span['id']: span['value'] for span in affected_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        dir_map = {span['id']: span['value'] for span in direction_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        
        # Convert gold relations from span IDs to actual values
        gold_rels = set()
        for r in gold_rels_raw:
            if isinstance(r, dict) and 'route_span_id' in r and 'direction_span_id' in r:
                route_id = r['route_span_id']
                dir_id = r['direction_span_id']
                if route_id in route_map and dir_id in dir_map:
                    gold_rels.add((route_map[route_id], dir_map[dir_id]))
        
        # Overall metrics
        overall['tp'] += len(llm_rels & gold_rels)
        overall['fp'] += len(llm_rels - gold_rels)
        overall['fn'] += len(gold_rels - llm_rels)
        
        # Determine complexity
        routes_gold = safe_json_loads(row['routes_gold'])
        directions_gold_raw = safe_json_loads(row['directions_gold'])
        directions_gold = [
            d['text'] if isinstance(d, dict) and 'text' in d else d
            for d in directions_gold_raw
        ]
        num_routes = len(set(routes_gold))
        num_directions = len(set(directions_gold))
        total_entities = num_routes + num_directions
        
        if total_entities <= 2:
            complexity = 'simple'
        elif total_entities <= 6:
            complexity = 'moderate'
        else:
            complexity = 'complex'
        
        by_complexity[complexity]['tp'] += len(llm_rels & gold_rels)
        by_complexity[complexity]['fp'] += len(llm_rels - gold_rels)
        by_complexity[complexity]['fn'] += len(gold_rels - llm_rels)
    
    return {
        'overall': calculate_metrics(**overall),
        'by_complexity': {
            comp: calculate_metrics(**counts) 
            for comp, counts in by_complexity.items()
        }
    }


def generate_error_analysis(merged_df: pd.DataFrame) -> Dict[str, List]:
    """
    Identify and categorize common errors
    """
    errors = {
        'route_misses': [],
        'direction_misses': [],
        'relation_misses': [],
        'false_positives': []
    }
    
    for _, row in merged_df.iterrows():
        alert_id = row['alert_id']
        header = row['header']
        
        # Routes
        llm_routes = set(safe_json_loads(row['routes_llm']))
        gold_routes = set(safe_json_loads(row['routes_gold']))
        missed_routes = gold_routes - llm_routes
        
        if missed_routes:
            errors['route_misses'].append({
                'alert_id': alert_id,
                'header': header,
                'missed': list(missed_routes)
            })
        
        # Directions
        llm_dirs = set(
            d['text'] if isinstance(d, dict) else d 
            for d in safe_json_loads(row['directions_llm'])
        )
        gold_dirs = set(
            d['text'] if isinstance(d, dict) else d 
            for d in safe_json_loads(row['directions_gold'])
        )
        missed_dirs = gold_dirs - llm_dirs
        
        if missed_dirs:
            errors['direction_misses'].append({
                'alert_id': alert_id,
                'header': header,
                'missed': list(missed_dirs)
            })
        
        # Relations - deduplicate both LLM and gold relations before comparison
        llm_rels = set(
            (r['route'], r['direction']) 
            for r in safe_json_loads(row['relations_llm'])
            if isinstance(r, dict) and 'route' in r and 'direction' in r
        )
        
        # Convert gold relations using span-based data
        gold_rels_raw = safe_json_loads(row['relations_gold'])
        affected_spans = safe_json_loads(row['affected_spans_gold'])
        direction_spans = safe_json_loads(row['direction_spans_gold'])
        
        # Build mapping from span ID to value
        route_map = {span['id']: span['value'] for span in affected_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        dir_map = {span['id']: span['value'] for span in direction_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        
        # Convert gold relations from span IDs to actual values
        gold_rels = set()
        for r in gold_rels_raw:
            if isinstance(r, dict) and 'route_span_id' in r and 'direction_span_id' in r:
                route_id = r['route_span_id']
                dir_id = r['direction_span_id']
                if route_id in route_map and dir_id in dir_map:
                    gold_rels.add((route_map[route_id], dir_map[dir_id]))
        
        missed_rels = gold_rels - llm_rels
        
        if missed_rels:
            errors['relation_misses'].append({
                'alert_id': alert_id,
                'header': header,
                'missed': list(missed_rels)
            })
        
        # False positives
        fp_rels = llm_rels - gold_rels
        if fp_rels:
            errors['false_positives'].append({
                'alert_id': alert_id,
                'header': header,
                'false_positives': list(fp_rels)
            })
    
    return errors


def extract_misclassified_rows(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all rows with any misclassifications (routes, directions, or relations)
    Returns a DataFrame with error details
    """
    misclassified_rows = []
    
    for idx, row in merged_df.iterrows():
        has_error = False
        error_details = {
            'alert_id': row['alert_id'],
            'header': row['header'],
            'agency': row['agency'],
            'date': row.get('date', ''),
        }
        
        # Check routes
        llm_routes = set(safe_json_loads(row['routes_llm']))
        gold_routes = set(safe_json_loads(row['routes_gold']))
        
        routes_correct = llm_routes & gold_routes
        routes_missed = gold_routes - llm_routes
        routes_extra = llm_routes - gold_routes
        
        error_details['routes_gold'] = json.dumps(sorted(gold_routes))
        error_details['routes_llm'] = json.dumps(sorted(llm_routes))
        error_details['routes_correct'] = json.dumps(sorted(routes_correct))
        error_details['routes_missed'] = json.dumps(sorted(routes_missed))
        error_details['routes_extra'] = json.dumps(sorted(routes_extra))
        
        if routes_missed or routes_extra:
            has_error = True
        
        # Check directions
        llm_dirs = set(
            d['text'] if isinstance(d, dict) else d 
            for d in safe_json_loads(row['directions_llm'])
        )
        gold_dirs = set(
            d['text'] if isinstance(d, dict) else d 
            for d in safe_json_loads(row['directions_gold'])
        )
        
        dirs_correct = llm_dirs & gold_dirs
        dirs_missed = gold_dirs - llm_dirs
        dirs_extra = llm_dirs - gold_dirs
        
        error_details['directions_gold'] = json.dumps(sorted(gold_dirs))
        error_details['directions_llm'] = json.dumps(sorted(llm_dirs))
        error_details['directions_correct'] = json.dumps(sorted(dirs_correct))
        error_details['directions_missed'] = json.dumps(sorted(dirs_missed))
        error_details['directions_extra'] = json.dumps(sorted(dirs_extra))
        
        if dirs_missed or dirs_extra:
            has_error = True
        
        # Check relations
        llm_rels = set(
            (r['route'], r['direction']) 
            for r in safe_json_loads(row['relations_llm'])
            if isinstance(r, dict) and 'route' in r and 'direction' in r
        )
        
        # Convert gold relations using span-based data
        gold_rels_raw = safe_json_loads(row['relations_gold'])
        affected_spans = safe_json_loads(row['affected_spans_gold'])
        direction_spans = safe_json_loads(row['direction_spans_gold'])
        
        route_map = {span['id']: span['value'] for span in affected_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        dir_map = {span['id']: span['value'] for span in direction_spans if isinstance(span, dict) and 'id' in span and 'value' in span}
        
        gold_rels = set()
        for r in gold_rels_raw:
            if isinstance(r, dict) and 'route_span_id' in r and 'direction_span_id' in r:
                route_id = r['route_span_id']
                dir_id = r['direction_span_id']
                if route_id in route_map and dir_id in dir_map:
                    gold_rels.add((route_map[route_id], dir_map[dir_id]))
        
        rels_correct = llm_rels & gold_rels
        rels_missed = gold_rels - llm_rels
        rels_extra = llm_rels - gold_rels
        
        error_details['relations_gold'] = json.dumps([f"{r[0]}->{r[1]}" for r in sorted(gold_rels)])
        error_details['relations_llm'] = json.dumps([f"{r[0]}->{r[1]}" for r in sorted(llm_rels)])
        error_details['relations_correct'] = json.dumps([f"{r[0]}->{r[1]}" for r in sorted(rels_correct)])
        error_details['relations_missed'] = json.dumps([f"{r[0]}->{r[1]}" for r in sorted(rels_missed)])
        error_details['relations_extra'] = json.dumps([f"{r[0]}->{r[1]}" for r in sorted(rels_extra)])
        
        if rels_missed or rels_extra:
            has_error = True
        
        # Error flags
        error_details['has_route_error'] = bool(routes_missed or routes_extra)
        error_details['has_direction_error'] = bool(dirs_missed or dirs_extra)
        error_details['has_relation_error'] = bool(rels_missed or rels_extra)
        
        if has_error:
            misclassified_rows.append(error_details)
    
    return pd.DataFrame(misclassified_rows)


# ============================================================================
# Results Formatting and Output
# ============================================================================

def print_results(ner_results: Dict, rel_results: Dict, merged_df: pd.DataFrame):
    """Print formatted evaluation results"""
    output = []
    
    output.append("=" * 80)
    output.append("LLM ZERO-SHOT EVALUATION RESULTS")
    output.append("=" * 80)
    output.append(f"\nDataset: {GOLD_CSV}")
    output.append(f"LLM Predictions: {LLM_CSV}")
    output.append(f"Total Alerts Evaluated: {len(merged_df)}")
    output.append("")
    
    # ========== NER Results ==========
    output.append("=" * 80)
    output.append("NAMED ENTITY RECOGNITION (NER)")
    output.append("=" * 80)
    
    output.append("\n--- ROUTE EXTRACTION ---")
    r = ner_results['routes']
    output.append(f"Precision: {r['precision']:.4f}  (TP={r['tp']}, FP={r['fp']})")
    output.append(f"Recall:    {r['recall']:.4f}  (TP={r['tp']}, FN={r['fn']})")
    output.append(f"F1 Score:  {r['f1']:.4f}")
    
    output.append("\n--- DIRECTION EXTRACTION ---")
    d = ner_results['directions']
    output.append(f"Precision: {d['precision']:.4f}  (TP={d['tp']}, FP={d['fp']})")
    output.append(f"Recall:    {d['recall']:.4f}  (TP={d['tp']}, FN={d['fn']})")
    output.append(f"F1 Score:  {d['f1']:.4f}")
    
    if ner_results['direction_types']:
        output.append("\n--- DIRECTION BY TYPE ---")
        output.append(f"{'Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        output.append("-" * 60)
        for dtype, metrics in sorted(ner_results['direction_types'].items()):
            output.append(
                f"{dtype:<20} {metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}"
            )
    
    # ========== RE Results ==========
    output.append("\n" + "=" * 80)
    output.append("RELATION EXTRACTION (RE)")
    output.append("=" * 80)
    
    output.append("\n--- OVERALL PERFORMANCE ---")
    o = rel_results['overall']
    output.append(f"Precision: {o['precision']:.4f}  (TP={o['tp']}, FP={o['fp']})")
    output.append(f"Recall:    {o['recall']:.4f}  (TP={o['tp']}, FN={o['fn']})")
    output.append(f"F1 Score:  {o['f1']:.4f}")
    
    output.append("\n--- BY ALERT COMPLEXITY ---")
    output.append(f"{'Complexity':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<10}")
    output.append("-" * 65)
    for comp, metrics in rel_results['by_complexity'].items():
        count = metrics['tp'] + metrics['fn']
        output.append(
            f"{comp.capitalize():<15} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {count:<10}"
        )
    
    output.append("\n" + "=" * 80)
    output.append("SUMMARY")
    output.append("=" * 80)
    output.append(f"\nRoute NER F1:       {ner_results['routes']['f1']:.4f}")
    output.append(f"Direction NER F1:   {ner_results['directions']['f1']:.4f}")
    output.append(f"Relation RE F1:     {rel_results['overall']['f1']:.4f}")
    output.append("\n" + "=" * 80)
    
    # Print to console
    for line in output:
        print(line)
    
    # Save to file
    with open(OUTPUT_RESULTS, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"\nResults saved to: {OUTPUT_RESULTS}")


def print_error_samples(errors: Dict, max_samples: int = 5):
    """Print sample errors for qualitative analysis"""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS (Sample)")
    print("=" * 80)
    
    if errors['route_misses']:
        print(f"\n--- MISSED ROUTES (showing {min(max_samples, len(errors['route_misses']))} of {len(errors['route_misses'])}) ---")
        for err in errors['route_misses'][:max_samples]:
            print(f"Alert {err['alert_id']}: {err['header']}")
            print(f"  Missed: {err['missed']}\n")
    
    if errors['direction_misses']:
        print(f"\n--- MISSED DIRECTIONS (showing {min(max_samples, len(errors['direction_misses']))} of {len(errors['direction_misses'])}) ---")
        for err in errors['direction_misses'][:max_samples]:
            print(f"Alert {err['alert_id']}: {err['header']}")
            print(f"  Missed: {err['missed']}\n")
    
    if errors['relation_misses']:
        print(f"\n--- MISSED RELATIONS (showing {min(max_samples, len(errors['relation_misses']))} of {len(errors['relation_misses'])}) ---")
        for err in errors['relation_misses'][:max_samples]:
            print(f"Alert {err['alert_id']}: {err['header']}")
            print(f"  Missed: {err['missed']}\n")
    
    if errors['false_positives']:
        print(f"\n--- FALSE POSITIVES (showing {min(max_samples, len(errors['false_positives']))} of {len(errors['false_positives'])}) ---")
        for err in errors['false_positives'][:max_samples]:
            print(f"Alert {err['alert_id']}: {err['header']}")
            print(f"  FPs: {err['false_positives']}\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main evaluation pipeline"""
    print("Loading datasets...")
    
    # Load data
    try:
        gold_df = pd.read_csv(GOLD_CSV)
        llm_df = pd.read_csv(LLM_CSV, sep=';')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure {GOLD_CSV} and {LLM_CSV} are in the current directory.")
        return
    
    print(f"Loaded {len(gold_df)} gold annotations")
    print(f"Loaded {len(llm_df)} LLM predictions")
    
    # Rename columns to match expected format
    gold_df = gold_df.rename(columns={
        'affected': 'routes_gold',
        'direction_gold': 'directions_gold',
        'relations_gold': 'relations_gold'
    })
    
    llm_df = llm_df.rename(columns={
        'affected': 'routes_llm',
        'direction_llm': 'directions_llm',
        'relation_names_llm': 'relations_llm'
    })
    
    # Merge datasets
    merged_df = gold_df.merge(
        llm_df[['alert_id', 'routes_llm', 'directions_llm', 'relations_llm']], 
        on='alert_id',
        how='inner'
    )
    
    print(f"Merged {len(merged_df)} matching alerts")
    print(f"First 5 alert_ids from gold CSV: {gold_df['alert_id'].head(5).tolist()}")
    print(f"First 5 alert_ids after merge: {merged_df['alert_id'].head(5).tolist()}")
    print(f"Last row number (0-indexed): {len(merged_df) - 1}")
    print(f"Last row alert_id: {merged_df.iloc[-1]['alert_id']}")
    print(f"Last row header: {merged_df.iloc[-1]['header'][:80]}...\n")
    
    if len(merged_df) == 0:
        print("Error: No matching alert_ids found between datasets!")
        return
    
    # Run evaluations
    print("Evaluating NER performance...")
    ner_results = evaluate_ner(merged_df)
    
    print("Evaluating RE performance...")
    rel_results = evaluate_relations(merged_df)
    
    print("Generating error analysis...")
    errors = generate_error_analysis(merged_df)
    
    # Extract misclassified rows
    print("Extracting misclassified rows...")
    misclassified_df = extract_misclassified_rows(merged_df)
    
    # Save misclassified rows to CSV
    misclassified_df.to_csv(OUTPUT_MISCLASSIFIED, index=False, encoding='utf-8')
    print(f"Saved {len(misclassified_df)} misclassified rows to: {OUTPUT_MISCLASSIFIED}")
    
    # Print results
    print_results(ner_results, rel_results, merged_df)
    print_error_samples(errors, max_samples=5)
    
    print("\nEvaluation complete!")
    print(f"Misclassified rows exported to: {OUTPUT_MISCLASSIFIED}")


if __name__ == "__main__":
    main()
