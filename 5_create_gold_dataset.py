import pandas as pd
import json
import numpy as np
from collections import Counter

# Configuration
INPUT_FILE = 'Preprocessed/MTA_Data_silver_relations.csv'
OUTPUT_FILE = 'Preprocessed/MTA_Data_gold_dataset.csv'
TOTAL_SAMPLES = 600
RANDOM_SEED = 42

# Target distribution
COMPLEXITY_DISTRIBUTION = {
    'simple': 120,      # 20%: 2 directions
    'moderate': 240,    # 40%: 3-4 directions
    'complex': 240      # 40%: 4+ directions
}

def calculate_complexity(row):
    # Calculate complexity score based on number of directions only.
    try:
        num_directions = len(json.loads(row['direction_spans']))
        
        if num_directions == 2:
            return 'simple'
        elif num_directions in [3, 4]:
            return 'moderate'
        elif num_directions > 4:
            return 'complex'
        else:
            return None  # Exclude rows with less than 2 directions
    except:
        return None  # Default for parsing errors

def get_direction_types(row):
    # Extract all direction types from direction_spans.
    try:
        directions = json.loads(row['direction_spans'])
        return [d['value'] for d in directions]
    except:
        return []

def stratified_sample(df, target_dist, random_seed=42):
    # Perform stratified sampling based on complexity.
    # Within each complexity stratum, ensure direction type diversity.
    samples = []
    
    for complexity, n_samples in target_dist.items():
        # Filter by complexity
        stratum = df[df['complexity'] == complexity].copy()
        
        if len(stratum) < n_samples:
            print(f"Warning: Only {len(stratum)} samples available for '{complexity}', "
                  f"requested {n_samples}")
            samples.append(stratum)
        else:
            # Sample with priority for direction diversity
            # Get direction distribution in this stratum
            stratum['direction_types'] = stratum.apply(get_direction_types, axis=1)
            
            # Sample
            sampled = stratum.sample(n=n_samples, random_state=random_seed)
            samples.append(sampled)
            
            # Print stats for this stratum
            print(f"\n{complexity.upper()} complexity: {n_samples} samples")
            all_directions = [d for dirs in sampled['direction_types'] for d in dirs]
            direction_counts = Counter(all_directions)
            for direction, count in direction_counts.most_common(10):
                print(f"  {direction}: {count}")
    
    return pd.concat(samples, ignore_index=True)

def prepare_gold_dataset(df):
    # Prepare dataset for gold annotation
    df_gold = df.copy()
    
    # Columns to duplicate (silver to gold)
    annotation_columns = [
        'direction',
        'affected_spans',
        'direction_spans',
        'relation_names',
        'relations'
    ]
    
    # Rename original columns to *_silver
    rename_map = {col: f"{col}_silver" for col in annotation_columns}
    df_gold.rename(columns=rename_map, inplace=True)
    
    # Add empty *_gold columns (initialized with empty lists)
    for col in annotation_columns:
        df_gold[f"{col}_gold"] = '[]'
    
    # Reorder columns: keep metadata, then silver columns, then gold columns
    metadata_cols = [col for col in df_gold.columns 
                    if not (col.endswith('_silver') or col.endswith('_gold'))]
    
    silver_cols = [col for col in df_gold.columns if col.endswith('_silver')]
    gold_cols = [col for col in df_gold.columns if col.endswith('_gold')]
    
    # Interleave silver and gold columns for easier annotation
    ordered_cols = metadata_cols.copy()
    for col in annotation_columns:
        if f"{col}_silver" in df_gold.columns:
            ordered_cols.append(f"{col}_silver")
        if f"{col}_gold" in df_gold.columns:
            ordered_cols.append(f"{col}_gold")
    
    df_gold = df_gold[ordered_cols]
    
    # Add annotation status column
    df_gold.insert(0, 'annotation_status', 'pending')
    df_gold.insert(1, 'annotator_notes', '')
    
    return df_gold

def main():
    print("=" * 70)
    print("CREATING GOLD ANNOTATION SAMPLE")
    print("=" * 70)
    
    # Load silver dataset
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    
    # Calculate complexity for each row
    print("\nCalculating complexity scores...")
    df['complexity'] = df.apply(calculate_complexity, axis=1)
    
    # Filter out rows with less than 2 directions
    print(f"\nFiltering rows with at least 2 directions...")
    original_count = len(df)
    df = df[df['complexity'].notna()].copy()
    filtered_count = len(df)
    print(f"Kept {filtered_count:,} of {original_count:,} records ({100*filtered_count/original_count:.1f}%)")
    
    # Print complexity distribution
    print("\nComplexity distribution in full dataset:")
    for complexity, count in df['complexity'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {complexity}: {count:,} ({pct:.1f}%)")
    
    # Perform stratified sampling
    print(f"\nPerforming stratified sampling (n={TOTAL_SAMPLES})...")
    sampled_df = stratified_sample(df, COMPLEXITY_DISTRIBUTION, RANDOM_SEED)
    
    # Shuffle the final sample
    sampled_df = sampled_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Prepare for gold annotation
    print("\nPreparing gold annotation dataset...")
    gold_df = prepare_gold_dataset(sampled_df)
    
    # Remove temporary columns
    if 'complexity' in gold_df.columns:
        gold_df.drop(columns=['complexity'], inplace=True)
    if 'direction_types' in gold_df.columns:
        gold_df.drop(columns=['direction_types'], inplace=True)
    
    # Save to CSV
    print(f"\nSaving to {OUTPUT_FILE}...")
    gold_df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary statistics
    print("GOLD SAMPLE SUMMARY")
    print(f"Total samples: {len(gold_df):,}")
    print(f"\nColumns in gold dataset:")
    for i, col in enumerate(gold_df.columns, 1):
        print(f"  {i}. {col}")
    
    # Print some statistics on the sample
    print("\nSample statistics:")
    
    # Count routes and directions
    route_counts = []
    direction_counts = []
    relation_counts = []
    
    for _, row in gold_df.iterrows():
        try:
            route_counts.append(len(json.loads(row['affected_spans_silver'])))
            direction_counts.append(len(json.loads(row['direction_spans_silver'])))
            relation_counts.append(len(json.loads(row['relations_silver'])))
        except:
            pass
    
    print(f"  Avg routes per sample: {np.mean(route_counts):.2f}")
    print(f"  Avg directions per sample: {np.mean(direction_counts):.2f}")
    print(f"  Avg relations per sample: {np.mean(relation_counts):.2f}")
    
    print("\nRoutes distribution:")
    for i in range(1, min(6, max(route_counts) + 1)):
        count = sum(1 for x in route_counts if x == i)
        print(f"  {i} route(s): {count} samples")
    
    print("\nDirections distribution:")
    for i in range(1, min(6, max(direction_counts) + 1)):
        count = sum(1 for x in direction_counts if x == i)
        print(f"  {i} direction(s): {count} samples")
    
    print("Ready for gold annotation.")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
