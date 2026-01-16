import pandas as pd
import glob
import os

def process_bus_codes():
    """
    Extract route_short_name column from all routes files and combine into bus_codes.csv.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all routes files in the same directory
    pattern = os.path.join(script_dir, "routes *.txt")
    route_files = sorted(glob.glob(pattern))
    
    if not route_files:
        print("No routes files found.")
        return
    
    print(f"Found {len(route_files)} route files:")
    for file in route_files:
        print(f"  - {os.path.basename(file)}")
    
    # Collect all route_short_names
    all_route_short_names = []
    
    for file in route_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Check if route_short_name column exists
            if 'route_short_name' in df.columns:
                # Extract route_short_name column
                route_short_names = df['route_short_name'].tolist()
                all_route_short_names.extend(route_short_names)
                print(f"Extracted {len(route_short_names)} route_short_names from {os.path.basename(file)}")
            else:
                print(f"Warning: 'route_short_name' column not found in {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {e}")
    
    # Remove duplicates while preserving order
    unique_route_short_names = list(dict.fromkeys(all_route_short_names))
    
    # Create DataFrame with route_short_names
    result_df = pd.DataFrame({'route_short_name': unique_route_short_names})
    
    # Save to CSV in the same directory
    output_file = os.path.join(script_dir, "bus_codes.csv")
    result_df.to_csv(output_file, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total route_short_names collected: {len(all_route_short_names)}")
    print(f"Unique route_short_names: {len(unique_route_short_names)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    process_bus_codes()

