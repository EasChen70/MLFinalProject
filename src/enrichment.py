import pandas as pd
import os
import json

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

nhanes_processed = config.get('nhanes_processed')
faers_processed = config.get('faers_processed')
output_folder = config.get("output_folder", "processed")

# Load NHANES and FAERS datasets
try:
    nhanes_df = pd.read_csv(nhanes_processed)
    faers_df = pd.read_csv(faers_processed)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

