import pandas as pd
import csv

def extract_columns_from_csv(input_file, output_file=None):
    """
    Extract subject, relation, and object columns from CSV file
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        pandas.DataFrame: DataFrame with extracted columns
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Print original columns for verification
        print("Original columns:", df.columns.tolist())
        print(f"Original shape: {df.shape}")
        
        # Check if required columns exist
        required_columns = ['subject', 'relation', 'object']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # Try to find columns with similar names (case insensitive)
            available_cols = df.columns.tolist()
            for missing_col in missing_columns:
                for col in available_cols:
                    if missing_col.lower() in col.lower():
                        print(f"Found similar column '{col}' for '{missing_col}'")
        
        # Extract only the required columns
        extracted_df = df[required_columns].copy()
        
        # Display first few rows
        print("\nExtracted data (first 5 rows):")
        print(extracted_df.head())
        print(f"\nExtracted shape: {extracted_df.shape}")
        
        # Save to new CSV file if output_file is specified
        if output_file:
            extracted_df.to_csv(output_file, index=False)
            print(f"\nExtracted data saved to: {output_file}")
        
        return extracted_df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file.")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def extract_columns_manual_parsing(input_file, output_file=None):
    """
    Alternative method using manual CSV parsing for problematic files
    """
    try:
        extracted_data = []
        
        with open(input_file, 'r', encoding='utf-8') as file:
            # Read first line to get headers
            first_line = file.readline().strip()
            headers = first_line.split(',')
            
            print("Detected headers:", headers)
            
            # Find indices of required columns
            try:
                subject_idx = headers.index('subject')
                relation_idx = headers.index('relation')
                object_idx = headers.index('object')
            except ValueError as e:
                print(f"Error finding column indices: {e}")
                print("Available headers:", headers)
                return None
            
            # Read data rows
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) > max(subject_idx, relation_idx, object_idx):
                    extracted_row = {
                        'subject': row[subject_idx].strip(),
                        'relation': row[relation_idx].strip(),
                        'object': row[object_idx].strip()
                    }
                    extracted_data.append(extracted_row)
        
        # Convert to DataFrame
        extracted_df = pd.DataFrame(extracted_data)
        
        print(f"\nExtracted {len(extracted_data)} rows")
        print("\nFirst 5 rows:")
        print(extracted_df.head())
        
        # Save if output file specified
        if output_file:
            extracted_df.to_csv(output_file, index=False)
            print(f"\nData saved to: {output_file}")
        
        return extracted_df
        
    except Exception as e:
        print(f"Error in manual parsing: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_csv = "/content/extracted_triples_20250921_222745.csv"
    output_csv = "extracted_columns.csv"
    
    print("=== CSV Column Extractor ===")
    print("Extracting subject, relation, and object columns...\n")
    
    # Try pandas method first
    result = extract_columns_from_csv(input_csv, output_csv)
    
    # If pandas fails, try manual parsing
    if result is None:
        print("\nTrying alternative parsing method...")
        result = extract_columns_manual_parsing(input_csv, output_csv)
    
    if result is not None:
        print("\n=== Extraction completed successfully! ===")
        print(f"Total rows extracted: {len(result)}")
    else:
        print("\n=== Extraction failed ===")

# Quick function for immediate use
def quick_extract(file_path):
    """Quick extraction function"""
    df = pd.read_csv(file_path)
    return df[['subject', 'relation', 'object']]
