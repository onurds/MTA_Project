import pandas as pd
import os
import json

def preprocess_mta_data():

    # Preprocessing the MTA data by filtering NYCT Bus and NYCT Subway records,
    # removing specific columns, and standardizing the Affected column format.

    # Define file paths
    input_file = 'Data/MTA_Data.csv'
    output_dir = 'Preprocessed'
    output_file = os.path.join(output_dir, 'MTA_Data_preprocessed.csv')
    
    # Reading the CSV file
    df = pd.read_csv(input_file)
    print(f"Initial data shape: {df.shape}")
    
    # Filter to keep only NYCT Bus and NYCT Subway records
    print("Filtering for NYCT Bus and NYCT Subway records...")
    df = df[df['Agency'].isin(['NYCT Bus', 'NYCT Subway'])]
    print(f"Data shape after filtering: {df.shape}")

    # Drop the specified columns to focus on the relevant data
    df.drop(columns=['Event ID', 'Update Number', 'Description'], inplace=True)
    
    # Remove rows with NaN values in Header column
    df = df[df['Header'].notna()]
    print(f"Data shape after removing NaN headers: {df.shape}")

    # Remove duplicate headers to avoid duplicate data
    df = df.drop_duplicates(subset=['Header'], keep='first')
    print(f"Data shape after removing duplicates: {df.shape}")
    
    # Remove rows where Header column contains "this bus"
    df = df[~df['Header'].str.contains('this bus', case=False, na=False)]
    print(f"Data shape after removing 'this bus' rows: {df.shape}")
    
    # Remove rows where Header column contains "bound track" or "bound tracks"
    df = df[~df['Header'].str.contains(r'bound\s+tracks?', case=False, na=False, regex=True)]
    print(f"Data shape after removing 'bound track(s)' rows: {df.shape}")
    
    # Remove rows where Header has fewer than 3 words Or fewer than 8 characters, because it's too short to be a valid header
    df = df[(df['Header'].str.split().str.len() >= 3) | (df['Header'].str.len() >= 8)]
    print(f"Data shape after applying minimum length filter: {df.shape}")
    
    # Convert Affected column to JSON array format
    df['Affected'] = df['Affected'].apply(lambda x: json.dumps(x.split(' | ')) if pd.notna(x) else '[]')
    
    # Create output directory
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Export to CSV
    print(f"Exporting preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete, Final data shape is: {df.shape}")

if __name__ == "__main__":
    preprocess_mta_data()

