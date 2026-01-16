import pandas as pd
import re
import os

def extract_directions():
    """
    Extract all words containing 'bound' from the Header column of MTA_Data_preprocessed.csv
    and save them to directions.csv in the Data folder.
    Also create a file with occurrence counts.
    """
    # Define file paths
    input_file = "/Users/onurds/Downloads/student_NLP1/Preprocessed/MTA_Data_preprocessed.csv"
    output_file = "/Users/onurds/Downloads/student_NLP1/Data/directions.csv"
    counts_file = "/Users/onurds/Downloads/student_NLP1/Data/directions_counts.csv"
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}")
    
    # Check if Header column exists
    if 'Header' not in df.columns:
        print("Error: 'Header' column not found in the CSV file.")
        return
    
    # Extract all words containing 'bound' from the Header column
    bound_words = set()
    bound_words_counts = {}
    
    # Pattern to match words containing 'bound' (case-insensitive)
    # Matches words like: Southbound, South-bound, NORTHBOUND, etc.
    pattern = r'\b[\w-]*bound[\w-]*\b'
    
    for idx, header in enumerate(df['Header']):
        if pd.notna(header):  # Check if the value is not NaN
            # Find all matches in the header text
            matches = re.findall(pattern, str(header), re.IGNORECASE)
            for match in matches:
                bound_words.add(match)
                # Count occurrences
                if match in bound_words_counts:
                    bound_words_counts[match] += 1
                else:
                    bound_words_counts[match] = 1
    
    # Convert set to sorted list
    bound_words_list = sorted(list(bound_words))
    
    print(f"\nFound {len(bound_words_list)} unique direction words containing 'bound':")
    for word in bound_words_list:
        print(f"  - {word}")
    
    # Create DataFrame with the direction words
    result_df = pd.DataFrame({'direction': bound_words_list})
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")
    
    # Create DataFrame with counts sorted by occurrence (descending)
    counts_df = pd.DataFrame(list(bound_words_counts.items()), columns=['direction', 'count'])
    counts_df = counts_df.sort_values('count', ascending=False)
    
    # Save counts to CSV
    counts_df.to_csv(counts_file, index=False)
    print(f"Counts saved to: {counts_file}")
    
    # Show top 10 most frequent
    print(f"\nTop 10 most frequent direction words:")
    for idx, row in counts_df.head(10).iterrows():
        print(f"  {row['direction']}: {row['count']} occurrences")

if __name__ == "__main__":
    extract_directions()

