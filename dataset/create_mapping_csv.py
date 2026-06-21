import pandas as pd
from transcript_mapping_dict import transcript_mapping
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load source CSV files using paths relative to script location
# Note: These CSVs use semicolon as delimiter
chisco_part1 = pd.read_csv(os.path.join(script_dir, "src_sentences/chisco_part1.csv"), sep=";")
chisco_part2 = pd.read_csv(os.path.join(script_dir, "src_sentences/chisco_part2.csv"), sep=";")

# Map source files to dataframes
source_files = {
    "chisco_part1.csv": chisco_part1,
    "chisco_part2.csv": chisco_part2
}

# Create list to store results
output_rows = []

# Process each mapping entry
for mapping in transcript_mapping:
    start_id = mapping["start_id"]
    end_id = mapping["end_id"]
    src_file = mapping["src_file"]
    src_start_idx = mapping["src_start_idx"]
    src_end_idx = mapping["src_end_idx"]
    gender = mapping["gender"]
    
    # Extract the subject ID and number from start_id (e.g., "1_DAM" -> ("1", "DAM"))
    start_match = re.match(r"(\d+)_([A-Z]+)", start_id)
    end_match = re.match(r"(\d+)_([A-Z]+)", end_id)
    
    if start_match and end_match:
        start_num = int(start_match.group(1))
        subject = start_match.group(2)
        end_num = int(end_match.group(1))
        
        # Get the source dataframe
        src_df = source_files[src_file]
        
        # Extract sentences from source file (using the "Kalimat" column)
        sentences = src_df.iloc[src_start_idx:src_end_idx+1]["Kalimat"].values.tolist()
        
        # Create output rows
        for i, sentence in enumerate(sentences):
            current_id = f"{start_num + i}_{subject}"
            output_rows.append({
                "id": current_id,
                "subject": subject,
                "sentence": sentence,
                "gender": gender
            })

# Create output dataframe
output_df = pd.DataFrame(output_rows)

# Save to CSV
output_file = os.path.join(script_dir, "raw_transcript_mapping.csv")
output_df.to_csv(output_file, index=False)

print(f"Mapping saved to {output_file}")
print(f"Total rows: {len(output_df)}")
print("\nPreview:")
print(output_df.head(10))
