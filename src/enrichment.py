import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Step 1: Define Polypharmacy Risk
def assign_polypharmacy_risk(num_prescriptions):
    if num_prescriptions <= 4:
        return 'Low'
    elif 5 <= num_prescriptions <= 8:
        return 'Moderate'
    else:
        return 'High'

nhanes_df['polypharmacy_risk'] = nhanes_df['num_presc_taken'].apply(assign_polypharmacy_risk)

# Step 2: Calculate ADE Likelihood and Average Drug Risk Score from FAERS
faers_aggregates = faers_df.groupby(['age_group', 'polypharmacy_risk']).agg(
    ade_likelihood=('outc_cod', lambda x: x.str.contains('HO|OT', na=False).sum() / len(x)),
    avg_drug_risk_score=('drug_risk_score', 'mean'),
    confidence_indicator=('outc_cod', 'size')
).reset_index()

# Step 3: Merge Aggregated Features to NHANES
nhanes_df = pd.merge(nhanes_df, faers_aggregates, on=['age_group', 'polypharmacy_risk'], how='left')

# Step 4: Identify High-Risk Drugs and Adjust Enrichment
high_risk_drugs = faers_df[faers_df['drug_risk_score'] > 0.2]['drugname'].unique()
filtered_faers_df = faers_df[~faers_df['drugname'].isin(high_risk_drugs)]

filtered_aggregates = filtered_faers_df.groupby(['age_group', 'polypharmacy_risk']).agg(
    filtered_ade_likelihood=('outc_cod', lambda x: x.str.contains('HO|OT', na=False).sum() / len(x)),
    filtered_avg_drug_risk_score=('drug_risk_score', 'mean')
).reset_index()

# Merge recalculated values into NHANES
nhanes_df = pd.merge(nhanes_df, filtered_aggregates, on=['age_group', 'polypharmacy_risk'], how='left')

# Step 5: Normalize Features and Rebalance Composite Risk Score
scaler = MinMaxScaler()
normalized_columns = ['health_score', 'filtered_ade_likelihood', 'filtered_avg_drug_risk_score']
nhanes_df[[f'normalized_{col}' for col in normalized_columns]] = scaler.fit_transform(
    nhanes_df[normalized_columns]
)

# Calculate rebalanced composite risk score
nhanes_df['rebalanced_composite_risk_score'] = (
    0.5 * nhanes_df['normalized_health_score'] +
    0.3 * nhanes_df['normalized_filtered_ade_likelihood'] +
    0.2 * nhanes_df['normalized_filtered_avg_drug_risk_score']
)

# Step 6: Log Transform Composite Risk Score
nhanes_df['composite_risk_score_log'] = np.log1p(nhanes_df['rebalanced_composite_risk_score'])

# Retain only meaningful columns
columns_to_keep = [
    'age_group', 'ever_asthma', 'current_asthma', 'asthma_attack_past_year',
    'asthma_emergency_past_year', 'any_liver_condition', 'cancer_malignancy',
    'num_presc_taken', 'polypharmacy_risk', 'confidence_indicator',
    'normalized_health_score', 'normalized_filtered_ade_likelihood',
    'normalized_filtered_avg_drug_risk_score', 'rebalanced_composite_risk_score',
    'composite_risk_score_log'
]
nhanes_df = nhanes_df[columns_to_keep]

# Validation: Check for Missing Values
print("Missing Values by Column:")
print(nhanes_df.isna().sum())

# Validation: Check Feature Distributions
print("\nFeature Distributions:")
print(nhanes_df[['rebalanced_composite_risk_score', 'normalized_health_score',
                 'normalized_filtered_ade_likelihood', 'normalized_filtered_avg_drug_risk_score']].describe())

# Save Enriched Dataset
output_path = os.path.join(output_folder, 'enriched_nhanes_with_faers.csv')
os.makedirs(output_folder, exist_ok=True)
nhanes_df.to_csv(output_path, index=False)
print(f"\nEnriched NHANES dataset saved to {output_path}")
