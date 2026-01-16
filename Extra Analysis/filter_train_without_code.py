import pandas as pd
import re
import os

def filter_train_without_code():
    """
    Filter rows where Agency is 'NYCT Subway' and Header contains 'train' or 'trains'
    where NONE of the instances are preceded by train codes.
    
    A row is included only if ALL occurrences of 'train/trains' in the header
    lack a train code immediately before them.
    
    Train code format:
    1. Single letter A-Z
    2. Single digit 1-8
    3. Two-character combinations: letter-letter (e.g., GS) or letter-digit/digit-letter (e.g., 7X)
    4. Special codes: Shuttle, SIR
    
    Save the results to a CSV file.
    """
    # Define file paths
    input_file = "/Users/onurds/Downloads/student_NLP1/Data/MTA_Data.csv"
    output_file = "/Users/onurds/Downloads/student_NLP1/Data/train_without_code.csv"
    
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
    
    # Pattern for train codes:
    # - Single letter A-Z: [A-Z]
    # - Single digit 1-8: [1-8]
    # - Two letters: [A-Z]{2}
    # - Letter-digit or digit-letter: [A-Z][0-9] or [0-9][A-Z]
    # - Special codes: Shuttle, SIR
    train_code_pattern = r'\b([A-Z]|[1-8]|[A-Z]{2}|[A-Z][0-9]|[0-9][A-Z]|Shuttle|SIR)\b'
    
    # Pattern for train code followed by "train" or "trains"
    # This matches cases like "A train", "7 trains", "GS trains", etc.
    train_with_code_pattern = rf'{train_code_pattern}\s+trains?\b'
    
    def has_train_without_code(text):
        """
        Check if text contains 'train' or 'trains' where ALL instances are NOT preceded by a train code.
        Returns True only if the text has 'train/trains' and NONE of them are preceded by codes.
        """
        if pd.isna(text):
            return False
        
        text_str = str(text)
        
        # Find all instances of "train" or "trains"
        train_matches = list(re.finditer(r'\btrains?\b', text_str, re.IGNORECASE))
        
        if not train_matches:
            return False
        
        # For each "train/trains" match, check if it's preceded by a train code
        for match in train_matches:
            train_start = match.start()
            train_word = match.group()
            
            # Look at the text before "train/trains" to check for train code
            # Get a reasonable amount of context before (up to 20 characters)
            context_start = max(0, train_start - 20)
            context = text_str[context_start:train_start]
            
            # Check if the immediate preceding word is a train code
            # Pattern: train code followed by whitespace, then "train/trains"
            preceding_pattern = rf'{train_code_pattern}\s+$'
            
            if re.search(preceding_pattern, context, re.IGNORECASE):
                # This "train/trains" IS preceded by a train code
                # So this row should be excluded
                return False
        
        # All instances of "train/trains" are NOT preceded by codes
        return True
    
    # Apply filter
    df_train_without_code = df_subway[df_subway['Header'].apply(has_train_without_code)].copy()
    
    print(f"Rows where ALL 'train/trains' instances are NOT preceded by train code: {len(df_train_without_code)}")
    print(f"Percentage of subway rows: {(len(df_train_without_code) / len(df_subway) * 100):.2f}%")
    
    # Save to CSV
    df_train_without_code.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")
    
    # Display sample headers for inspection
    print(f"\nSample headers where ALL 'train/trains' are not preceded by train code (first 15):")
    sample_headers = df_train_without_code['Header'].dropna().head(15)
    for idx, header in enumerate(sample_headers, 1):
        # Highlight the problematic train/trains
        print(f"  {idx}. {header}")
    
    # Show distribution of Status Labels for these rows
    if 'Status Label' in df_train_without_code.columns:
        print(f"\nTop 10 Status Labels for rows where ALL 'train/trains' are not preceded by code:")
        status_counts = df_train_without_code['Status Label'].value_counts().head(10)
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    
    return df_train_without_code

if __name__ == "__main__":
    result_df = filter_train_without_code()

