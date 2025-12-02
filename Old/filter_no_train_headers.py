import pandas as pd
import re
import os

def filter_no_train_headers():
    """
    Filter rows where Agency is 'NYCT Subway' and Header column does not contain
    'train' or 'trains' keywords.
    Save the results to a CSV file.
    """
    # Define file paths
    input_file = "/Users/onurds/Downloads/student_NLP1/Data/MTA_Data.csv"
    output_file = "/Users/onurds/Downloads/student_NLP1/Data/no_train_headers.csv"
    
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
    
    # Filter for headers that do NOT contain 'train' or 'trains' (case-insensitive)
    # Create a mask for rows that do NOT contain train/trains
    mask = df_subway['Header'].apply(
        lambda x: not bool(re.search(r'\btrain(?:s)?\b', str(x), re.IGNORECASE)) if pd.notna(x) else True
    )
    
    df_no_train = df_subway[mask].copy()
    
    print(f"Rows without 'train' or 'trains' in Header: {len(df_no_train)}")
    print(f"Percentage: {(len(df_no_train) / len(df_subway) * 100):.2f}%")
    
    # Save to CSV
    df_no_train.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")
    
    # Display some sample headers for inspection
    print(f"\nSample headers without 'train' or 'trains' (first 10):")
    sample_headers = df_no_train['Header'].dropna().head(10)
    for idx, header in enumerate(sample_headers, 1):
        print(f"  {idx}. {header}")
    
    # Show distribution of Status Labels for these rows
    if 'Status Label' in df_no_train.columns:
        print(f"\nStatus Label distribution for rows without 'train/trains':")
        status_counts = df_no_train['Status Label'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    
    return df_no_train

if __name__ == "__main__":
    result_df = filter_no_train_headers()

