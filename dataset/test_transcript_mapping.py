import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the transcript mapping CSV
csv_file = os.path.join(script_dir, "raw_transcript_mapping.csv")
df = pd.read_csv(csv_file)

# 1. Dataset shape
print(f"\n1. Dataset Shape:")
print(f"   Total rows: {len(df)}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Columns: {', '.join(df.columns.tolist())}")

# 2. Gender distribution
print(f"\n2. Gender Distribution:")
gender_dist = df['gender'].value_counts()
for gender, count in gender_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   {gender}: {count} ({percentage:.1f}%)")

# 3. Subject analysis
print(f"\n3. Subject Analysis:")
subject_dist = df['subject'].value_counts().sort_index()
print(f"   Total unique subjects: {len(subject_dist)}")
for subject, count in subject_dist.items():
    print(f"   {subject}: {count} sentences")