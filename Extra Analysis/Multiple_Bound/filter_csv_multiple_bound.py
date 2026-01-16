import csv
import re

def filter_csv_multiple_bound(input_file, output_file):
    """
    Filter CSV file to keep rows where Header column contains:
    1. At least 2 occurrences of 'bound', OR
    2. At least 1 occurrence of 'bound' AND at least 1 occurrence of 'both directions' or 'either direction'
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    rows_kept = 0
    rows_total = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read and write header
        header = next(reader)
        writer.writerow(header)
        
        # Find the index of the "Header" column
        try:
            header_col_index = header.index('Header')
        except ValueError:
            print("Error: 'Header' column not found in CSV file")
            return
        
        # Process each row
        for row in reader:
            rows_total += 1
            
            # Check if row has enough columns
            if len(row) <= header_col_index:
                continue
            
            header_text = row[header_col_index].lower()
            
            # Count occurrences of 'bound' (case-insensitive)
            bound_count = header_text.count('bound')
            
            # Check for directional keywords
            has_both_directions = 'both directions' in header_text
            has_either_direction = 'either direction' in header_text
            
            # Keep row if:
            # 1. 'bound' appears at least 2 times, OR
            # 2. 'bound' appears at least 1 time AND ('both directions' OR 'either direction' present)
            if bound_count >= 2 or (bound_count >= 1 and (has_both_directions or has_either_direction)):
                writer.writerow(row)
                rows_kept += 1
    
    print(f"Processing complete!")
    print(f"Total rows processed: {rows_total:,}")
    print(f"Rows kept: {rows_kept:,}")
    print(f"Rows filtered out: {rows_total - rows_kept:,}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    input_file = "/Users/onurds/Downloads/student_NLP1/Preprocessed/MTA_Data_preprocessed.csv"
    output_file = "/Users/onurds/Downloads/student_NLP1/Multiple_Bound/MTA_Service_Alerts_multiple_bound.csv"
    
    filter_csv_multiple_bound(input_file, output_file)

