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

# Apply filter to include only Asian/Pacific Islander participants
asian_ethnicities = [6]  # Assuming '6' corresponds to 'Non-Hispanic Asian' in RIDRETH3
nhanesdemo_df = nhanesdemo_df[nhanesdemo_df['ethnicity_detailed'].isin(asian_ethnicities)]

#Relevant medical conditions to drug events
relevant_medcon_columns = [
    'SEQN', 'MCQ010', 'MCQ035', 'MCQ040', 'MCQ050', 'MCQ160a', 'MCQ160b', 'MCQ160c', 
    'MCQ160d', 'MCQ160e', 'MCQ160f', 'MCQ160p', 'MCQ160l', 'MCQ500', 
    'MCQ510a', 'MCQ510c', 'MCQ220'
]
existing_medcon_columns = [col for col in relevant_medcon_columns if col in nhanesmedcon_df.columns]

nhanesmedcon_df = nhanesmedcon_df[existing_medcon_columns]
nhanesmedcon_df.rename(columns={'SEQN': 'participant_id'}, inplace=True)

#Rename columns for medcon
column_rename_mapping = {
    'MCQ010': 'ever_asthma',
    'MCQ035': 'current_asthma',
    'MCQ040': 'asthma_attack_past_year',
    'MCQ050': 'asthma_emergency_past_year',
    'MCQ160a': 'arthritis',
    'MCQ160b': 'congestive_heart_failure',
    'MCQ160c': 'coronary_heart_disease',
    'MCQ160d': 'angina',
    'MCQ160e': 'heart_attack',
    'MCQ160f': 'stroke',
    'MCQ160p': 'copd_emphysema',
    'MCQ160l': 'liver_condition',
    'MCQ500': 'any_liver_condition',
    'MCQ510a': 'fatty_liver',
    'MCQ510c': 'liver_cirrhosis',
    'MCQ220': 'cancer_malignancy'
}
existing_column_rename_mapping = {k: v for k, v in column_rename_mapping.items() if k in nhanesmedcon_df.columns}
nhanesmedcon_df.rename(columns=existing_column_rename_mapping, inplace=True)


#ID, DrugName
nhanesprescr_df = nhanesprescr_df[['SEQN', 'RXQ050']]
nhanesprescr_df.rename(columns={
    'SEQN': 'participant_id',
    'RXQ050': 'num_presc_taken'
}, inplace=True)

merged_df = nhanesdemo_df.merge(nhanesmedcon_df, on='participant_id', how='left')
merged_df = merged_df.merge(nhanesprescr_df, on='participant_id', how='left')

# Fill missing values for medical conditions with 0 (assumed absence of condition)
medical_condition_columns = list(existing_column_rename_mapping.values())
merged_df[medical_condition_columns] = merged_df[medical_condition_columns].fillna(0)

# Create health score by adding up all relevant medical conditions
condition_weights = {  # Assign weights to each condition
    'ever_asthma': 1, 'current_asthma': 1, 'asthma_attack_past_year': 1, 'asthma_emergency_past_year': 1,
    'arthritis': 1, 'congestive_heart_failure': 2, 'coronary_heart_disease': 2, 'angina': 2,
    'heart_attack': 3, 'stroke': 3, 'copd_emphysema': 2, 'liver_condition': 1,
    'any_liver_condition': 2, 'fatty_liver': 2, 'liver_cirrhosis': 3, 'cancer_malignancy': 3
}

# Initialize health score as 0
merged_df['health_score'] = 0

# Calculate health score based on conditions present
for condition, weight in condition_weights.items():
    if condition in merged_df.columns:
        merged_df['health_score'] += merged_df[condition].apply(lambda x: weight if x == 2 else 0)

# Fill missing values in prescription drugs with 'None' to indicate no prescription
merged_df['num_presc_taken'] = merged_df['num_presc_taken'].astype(object)
merged_df['num_presc_taken'] = merged_df['num_presc_taken'].fillna('None')

# Filter out rows where `num_presc_taken` is 'None'
merged_df = merged_df[merged_df['num_presc_taken'] != 'None']

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