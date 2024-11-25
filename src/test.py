import pandas as pd
import os
import json
import pyreadstat

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

nhanes_demo = config.get('nhanes_demo')
nhanes_medcon = config.get('nhanes_medcon')
nhanes_prescr = config.get('nhanes_prescr')
output_folder = config.get("output_folder", "processed")

#Load datasets
try:
    nhanesdemo_df, meta = pyreadstat.read_xport(nhanes_demo, encoding='ISO-8859-1')
    nhanesmedcon_df, meta = pyreadstat.read_xport(nhanes_medcon, encoding='ISO-8859-1')
    nhanesprescr_df, meta = pyreadstat.read_xport(nhanes_prescr, encoding='ISO-8859-1')
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}. Unable to read the file.")

#ID, Age, Ethnicity
nhanesdemo_df = nhanesdemo_df[['SEQN', 'RIDAGEYR', 'RIDRETH3']]
nhanesdemo_df.rename(columns={
    'SEQN': 'participant_id', 
    'RIDAGEYR': 'age', 
    'RIDRETH3': 'ethnicity_detailed'
}, inplace=True)

nhanesmedcon_df.rename(columns={'SEQN': 'participant_id'}, inplace=True)

#ID, DrugName
nhanesprescr_df = nhanesprescr_df[['SEQN', 'RXQ050']]
nhanesprescr_df.rename(columns={
    'SEQN': 'participant_id',
    'RXQ050': 'drug_name'
}, inplace=True)

merged_df = nhanesdemo_df.merge(nhanesmedcon_df, on='participant_id', how='left')
merged_df = merged_df.merge(nhanesprescr_df, on='participant_id', how='left')

# Fill missing values for medical conditions with 0 (assumed absence of condition)
medical_condition_columns = [col for col in merged_df.columns if col.startswith('MCQ')]
merged_df[medical_condition_columns] = merged_df[medical_condition_columns].fillna(0)

# Fill missing values in prescription drugs with 'None' to indicate no prescription
merged_df['drug_name'] = merged_df['drug_name'].astype(object)
merged_df['drug_name'] = merged_df['drug_name'].fillna('None')

#Turn age to integer
merged_df['age'] = merged_df['age'].astype(int)

#Ethnicity mapping
ethnicity_mapping = {
    1: 'Mexican-American',
    2: 'Other-Hispanic',
    3: 'Non-Hispanic White',
    4: 'Non-Hispanic Black',
    6: 'Non-Hispanic Asian',
    7: 'Other/Multi-Racial',
    '.': "Missing"
}
merged_df['ethnicity_detailed'] = merged_df['ethnicity_detailed'].map(ethnicity_mapping)

#drop duplicates
merged_df.drop_duplicates(subset='participant_id', inplace=True)


#Save the merged dataset
output_path = os.path.join(output_folder, 'merged_nhanes.csv')
os.makedirs(output_folder, exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"\nMerged NHANES dataset saved to {output_path}")