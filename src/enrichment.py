import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load configuration from JSON file
# This configuration provides paths to the NHANES and FAERS datasets
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

nhanes_processed = config.get('nhanes_processed')  # Path to processed NHANES data
faers_processed = config.get('faers_processed')    # Path to processed FAERS data
output_folder = config.get("output_folder", "processed")  # Output folder for the enriched dataset

# Load NHANES and FAERS datasets
try:
    nhanes_df = pd.read_csv(nhanes_processed)  # NHANES provides demographic and health-related data
    faers_df = pd.read_csv(faers_processed)   # FAERS provides adverse drug event reports
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Step 1: Define Polypharmacy Risk
# Polypharmacy risk is categorized based on the number of prescriptions taken
def assign_polypharmacy_risk(num_prescriptions):
    if num_prescriptions <= 4:
        return 'Low'  # Low risk for fewer prescriptions
    elif 5 <= num_prescriptions <= 8:
        return 'Moderate'  # Moderate risk for intermediate prescription count
    else:
        return 'High'  # High risk for a large number of prescriptions

nhanes_df['polypharmacy_risk'] = nhanes_df['num_presc_taken'].apply(assign_polypharmacy_risk)

# Step 2: Calculate ADE Likelihood and Average Drug Risk Score from FAERS
# Aggregate FAERS data by age group and polypharmacy risk to calculate:
# - Likelihood of adverse drug events (ADEs)
# - Average drug risk scores
# - Confidence indicator (number of reports in each group)
faers_aggregates = faers_df.groupby(['age_group', 'polypharmacy_risk']).agg(
    ade_likelihood=('outc_cod', lambda x: x.str.contains('HO|OT', na=False).sum() / len(x)),  # ADE ratio
    avg_drug_risk=('drug_risk_score', 'mean'),  # Average drug risk score
    confidence=('outc_cod', 'size')  # Number of reports in the group
).reset_index()

# Step 3: Merge Aggregated Features to NHANES
# Integrate FAERS-derived features (ADE likelihood, average drug risk, and confidence) into NHANES
nhanes_df = pd.merge(nhanes_df, faers_aggregates, on=['age_group', 'polypharmacy_risk'], how='left')

# Step 4: Identify High-Risk Drugs and Adjust Enrichment
# High-risk drugs are identified based on their drug risk scores (e.g., score > 0.2)
high_risk_drugs = faers_df[faers_df['drug_risk_score'] > 0.2]['drugname'].unique()
filtered_faers_df = faers_df[~faers_df['drugname'].isin(high_risk_drugs)]  # Exclude high-risk drugs

# Recalculate ADE likelihood and average drug risk score after removing high-risk drugs
filtered_aggregates = filtered_faers_df.groupby(['age_group', 'polypharmacy_risk']).agg(
    ade_likelihood=('outc_cod', lambda x: x.str.contains('HO|OT', na=False).sum() / len(x)),
    avg_drug_risk=('drug_risk_score', 'mean')
).reset_index()

# Merge recalculated features into NHANES
nhanes_df = pd.merge(nhanes_df, filtered_aggregates, on=['age_group', 'polypharmacy_risk'], how='left', suffixes=('', '_adjusted'))

# Step 5: Normalize Features
# Normalize selected columns (scaled between 0 and 1) for fair comparison and integration
scaler = MinMaxScaler()
columns_to_normalize = ['health_score', 'ade_likelihood_adjusted', 'avg_drug_risk_adjusted']
normalized_columns = [col + '_normalized' for col in columns_to_normalize]

# Perform normalization
nhanes_df[normalized_columns] = scaler.fit_transform(nhanes_df[columns_to_normalize])

# Rename normalized columns for easier reference
nhanes_df.rename(columns={
    'ade_likelihood_adjusted_normalized': 'ade_likelihood_normalized',
    'avg_drug_risk_adjusted_normalized': 'avg_drug_risk_normalized'
}, inplace=True)

# Weight FAERS metrics by confidence
nhanes_df['weighted_ade_likelihood'] = (
    nhanes_df['ade_likelihood_normalized'] * 
    (np.log1p(nhanes_df['confidence']) / np.log1p(nhanes_df['confidence'].max()))
)
nhanes_df['weighted_avg_drug_risk'] = (
    nhanes_df['avg_drug_risk_normalized'] * 
    (np.log1p(nhanes_df['confidence']) / np.log1p(nhanes_df['confidence'].max()))
)

# Calculate synthetic risk based on normalized and weighted features
def calculate_synthetic_risk(row):
    score = (
        0.3 * row['health_score_normalized'] +
        0.4 * row['weighted_ade_likelihood'] +
        0.3 * row['weighted_avg_drug_risk']
    )
    if score < 0.33:
        return 'Low Risk'
    elif score < 0.66:
        return 'Moderate Risk'
    else:
        return 'High Risk'

nhanes_df['synthetic_risk'] = nhanes_df.apply(calculate_synthetic_risk, axis=1)
nhanes_df['synthetic_risk'] = nhanes_df['synthetic_risk'].str.strip().str.title()

# Retain only meaningful columns for analysis and modeling
columns_to_keep = [
    'age_group',                   # Demographic group
    'polypharmacy_risk',           # Polypharmacy category
    'confidence',                  # Reliability indicator from FAERS
    'health_score_normalized',     # Normalized health score
    'ade_likelihood_normalized',   # Normalized ADE likelihood from FAERS
    'avg_drug_risk_normalized',    # Normalized average drug risk score
    'weighted_ade_likelihood',     # Weighted ADE likelihood by confidence
    'weighted_avg_drug_risk',      # Weighted average drug risk score by confidence
    'synthetic_risk'               # Synthetic risk category (target variable)
]
nhanes_df = nhanes_df[columns_to_keep]

# Save enriched dataset
output_path = os.path.join(output_folder, 'enriched_nhanes_with_faers.csv')
os.makedirs(output_folder, exist_ok=True)
nhanes_df.to_csv(output_path, index=False)
print(f"\nEnriched NHANES dataset saved to {output_path}")


def add_noise(df, noise_level=0.10):
    noisy_df = df.copy()
    numeric_features = ['health_score_normalized', 'ade_likelihood_normalized', 'avg_drug_risk_normalized', 'weighted_avg_drug_risk']
    
    for feature in numeric_features:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=len(noisy_df))
        noisy_df[feature] += noise
        
        # Clip values to keep them within bounds
        noisy_df[feature] = noisy_df[feature].clip(lower=0, upper=1)
    
    return noisy_df

# Oversample training dataset
low_risk = nhanes_df[nhanes_df['synthetic_risk'] == 'Low Risk']
moderate_risk = nhanes_df[nhanes_df['synthetic_risk'] == 'Moderate Risk']
high_risk = nhanes_df[nhanes_df['synthetic_risk'] == 'High Risk']


# Oversample Low Risk and High Risk to match Moderate Risk
low_risk_duplicated = add_noise(low_risk.sample(n=len(moderate_risk), replace=True, random_state=42))
high_risk_duplicated = add_noise(high_risk.sample(n=len(moderate_risk), replace=True, random_state=42))

# Combine into a balanced training dataset
balanced_training_df = pd.concat([low_risk_duplicated, high_risk_duplicated, moderate_risk])

# Recalculate synthetic risk for the oversampled dataset
balanced_training_df['synthetic_risk'] = balanced_training_df.apply(calculate_synthetic_risk, axis=1)

# Save oversampled training dataset
oversampled_output_path = os.path.join(output_folder, 'oversampled_training_nhanes.csv')
balanced_training_df.to_csv(oversampled_output_path, index=False)
print(f"Oversampled training dataset saved to {oversampled_output_path}")



