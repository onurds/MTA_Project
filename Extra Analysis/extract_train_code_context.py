import pandas as pd
import re
import os

def extract_train_code_context():
    """
    Extract train codes with their surrounding context (preceding and/or following words).
    
    Identifies cases where train codes might be misidentified (e.g., street names like "E 4 Av").
    
    Train code format:
    1. Single letter A-Z
    2. Single digit 1-8
    3. Two-character combinations: letter-letter (e.g., GS) or letter-digit/digit-letter (e.g., 7X)
    4. Special codes: Shuttle, SIR
    
    Excludes:
    - Words containing "bound" (e.g., Northbound, Queens-bound)
    - The word "trains" or "train"
    
    Creates a count list of code + context patterns.
    """
    # Define file paths
    input_file = "/Users/onurds/Downloads/student_NLP1/Data/MTA_Data.csv"
    output_file = "/Users/onurds/Downloads/student_NLP1/Data/train_code_context.csv"
    output_counts_file = "/Users/onurds/Downloads/student_NLP1/Data/train_code_context_counts.csv"
    output_details_file = "/Users/onurds/Downloads/student_NLP1/Data/train_code_context_details.csv"
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}")
    
    # Check if required columns exist
    if 'Agency' not in df.columns:
        print("Error: 'Agency' column not found in the CSV file.")
        return
    if 'Header' not in df.columns:
        print("Error: 'Header' column not found in the CSV file.")
        return
    
    # Filter for NYCT Subway agency
    df_subway = df[df['Agency'] == 'NYCT Subway'].copy()
    print(f"Rows with Agency 'NYCT Subway': {len(df_subway)}")
    
    # Pattern for train codes
    train_code_pattern = r'\b([A-Z]|[1-8]|[A-Z]{2}|[A-Z][0-9]|[0-9][A-Z]|Shuttle|SIR)\b'
    
    # List of valid train codes for checking
    def is_train_code(word):
        """Check if a word is a train code."""
        if not word:
            return False
        return bool(re.fullmatch(train_code_pattern, word))
    
    # Store all extracted contexts with full header information
    context_patterns = []
    context_details = []  # Store detailed information with Alert ID and full Header
    
    def extract_contexts_from_text(text):
        """
        Extract train codes with their context from text.
        Returns a list of context strings.
        """
        if pd.isna(text):
            return []
        
        text_str = str(text)
        contexts = []
        
        # Find all train codes
        for match in re.finditer(train_code_pattern, text_str):
            code = match.group()
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract preceding word (if exists)
            # Look backwards from the code
            before_text = text_str[:start_pos]
            # Get the last word before the code (word characters, hyphens, or single quote)
            preceding_match = re.search(r'([\w\-\']+)\s*$', before_text)
            preceding_word = preceding_match.group(1) if preceding_match else None
            
            # Extract following word (if exists)
            # Look forward from the code
            after_text = text_str[end_pos:]
            # Get the first word after the code
            following_match = re.search(r'^\s*([\w\-\']+)', after_text)
            following_word = following_match.group(1) if following_match else None
            
            # Filter out excluded words
            # Exclude words containing "bound" or being "train"/"trains"
            if preceding_word:
                if 'bound' in preceding_word.lower() or preceding_word.lower() in ['train', 'trains']:
                    preceding_word = None
            
            if following_word:
                if 'bound' in following_word.lower() or following_word.lower() in ['train', 'trains']:
                    following_word = None
            
            # Skip if both preceding and following words are train codes
            # (these are just train line listings like "2 3" or "E F")
            if preceding_word and following_word:
                if is_train_code(preceding_word) and is_train_code(following_word):
                    continue
            
            # Skip if only adjacent word is a train code and no other context
            # This handles cases like just "E F" with nothing else
            if preceding_word and not following_word:
                if is_train_code(preceding_word):
                    continue
            if following_word and not preceding_word:
                if is_train_code(following_word):
                    continue
            
            # Build context string
            # Include code with preceding and/or following words
            if preceding_word or following_word:
                parts = []
                if preceding_word:
                    parts.append(preceding_word)
                parts.append(code)
                if following_word:
                    parts.append(following_word)
                
                context_str = ' '.join(parts)
                contexts.append(context_str)
        
        return contexts
    
    # Extract all contexts
    print("\nExtracting train code contexts...")
    for idx, row in df_subway.iterrows():
        header = row['Header']
        alert_id = row.get('Alert ID', '')
        contexts = extract_contexts_from_text(header)
        context_patterns.extend(contexts)
        
        # Store details with full header information
        for context in contexts:
            context_details.append({
                'alert_id': alert_id,
                'context': context,
                'full_header': header
            })
    
    print(f"Total contexts extracted: {len(context_patterns)}")
    
    # Count occurrences
    from collections import Counter
    context_counts = Counter(context_patterns)
    
    print(f"Unique context patterns: {len(context_counts)}")
    
    # Create DataFrames
    # All contexts
    contexts_df = pd.DataFrame({'context': context_patterns})
    contexts_df.to_csv(output_file, index=False)
    print(f"\nAll contexts saved to: {output_file}")
    
    # Counts sorted by frequency
    counts_df = pd.DataFrame(context_counts.items(), columns=['context', 'count'])
    counts_df = counts_df.sort_values('count', ascending=False)
    counts_df.to_csv(output_counts_file, index=False)
    print(f"Context counts saved to: {output_counts_file}")
    
    # Details with full headers
    details_df = pd.DataFrame(context_details)
    details_df = details_df.sort_values('context')
    details_df.to_csv(output_details_file, index=False)
    print(f"Detailed contexts with full headers saved to: {output_details_file}")
    
    # Show top 30 most frequent contexts
    print(f"\nTop 30 most frequent train code contexts:")
    for idx, row in counts_df.head(30).iterrows():
        print(f"  {row['context']}: {row['count']} occurrences")
    
    # Show some examples that might be street names (containing "St", "Av", "Ave")
    print(f"\nPotential street name contexts (top 20):")
    street_pattern = counts_df[counts_df['context'].str.contains(r'\b(St|Av|Ave|Street|Avenue)\b', case=False, regex=True)]
    for idx, row in street_pattern.head(20).iterrows():
        print(f"  {row['context']}: {row['count']} occurrences")
    
    return counts_df

if __name__ == "__main__":
    result_df = extract_train_code_context()

