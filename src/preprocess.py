import pandas as pd
import os
import json

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Define data folder path and output folder
data_folder = config.get("data_folder")
output_folder = config.get("output_folder", "processed")
log_file = config.get("log_file", "faers_processing.log")

# Lists to hold data frames for each file type
demo_file = []
drug_file = []
reac_file = []
outc_file = []

# Function to load data from a file
def load_file(filepath):
    return pd.read_csv(filepath, delimiter='$', dtype=str)

# Load data from the specific quarter folder
quarter_id = "2024Q3"
ascii_folder_path = f"ASCII_{quarter_id}"
quarter_path = os.path.join(data_folder, ascii_folder_path)

if os.path.isdir(quarter_path):
    # Specify file path
    demo_path = os.path.join(quarter_path, 'DEMO24Q3.txt')
    drug_path = os.path.join(quarter_path, 'DRUG24Q3.txt')
    reac_path = os.path.join(quarter_path, 'REAC24Q3.txt')
    outc_path = os.path.join(quarter_path, 'OUTC24Q3.txt')

    # Load files
    if os.path.isfile(demo_path):
        demo_file.append(load_file(demo_path))
    if os.path.isfile(drug_path):
        drug_file.append(load_file(drug_path))
    if os.path.isfile(reac_path):
        reac_file.append(load_file(reac_path))
    if os.path.isfile(outc_path):
        outc_file.append(load_file(outc_path))

# Concatenate DataFrames for each type 
if demo_file:
    demo_df = pd.concat(demo_file, ignore_index=True)

    #focus on initial adverse drug event only
    demo_df = demo_df[demo_df['i_f_code'] == 'I']

    #filter for healthcare professionals only
    demo_df = demo_df[demo_df['occp_cod'].isin(['MD', 'PH', 'OT'])]

    # Extract year from 'event_dt' and modify the column directly
    if 'event_dt' in demo_df.columns:
        def extract_year(value):
            if len(value) == 8:
                return value[:4]  # YYYYMMDD
            elif len(value) == 6:
                return value[:4]  # YYYYMM
            elif len(value) == 4:
                return value  # YYYY
            else:
                return None  # Invalid value
        
        demo_df['event_dt'] = demo_df['event_dt'].apply(lambda x: extract_year(x) if pd.notnull(x) else None)
        # Drop rows with invalid or missing year values
        demo_df = demo_df.dropna(subset=['event_dt'])
        # Convert to numeric for easier handling
        demo_df['event_dt'] = pd.to_numeric(demo_df['event_dt'], errors='coerce')
        # Drop rows where event_dt conversion failed
        demo_df = demo_df.dropna(subset=['event_dt'])

    if 'rept_dt' in demo_df.columns:
        demo_df['rept_dt'] = demo_df['rept_dt'].apply(lambda x: extract_year(x) if pd.notnull(x) else None)
        # Drop rows with invalid or missing year values
        demo_df = demo_df.dropna(subset=['rept_dt'])
        # Convert to numeric for easier handling
        demo_df['rept_dt'] = pd.to_numeric(demo_df['rept_dt'], errors='coerce')
        # Drop rows where rept_dt conversion failed
        demo_df = demo_df.dropna(subset=['rept_dt'])

    # Drop rows with missing 'age' values
    demo_df = demo_df.dropna(subset=['age'])
    demo_df['age'] = pd.to_numeric(demo_df['age'], errors='coerce')
    demo_df = demo_df.dropna(subset=['age'])  # Drop rows with non-numeric or missing ages

    # Convert 'age' to age groups
    def convert_to_age_group(age):
        if age < 18:
            return 'Child'
        elif 18 <= age <= 44:
            return 'Young Adult'
        elif 45 <= age <= 64:
            return 'Middle Aged'
        else:
            return 'Elderly'

    demo_df['age'] = demo_df['age'].apply(convert_to_age_group)  
    #rename
    demo_df.rename(columns={'age': 'age_group'}, inplace=True)

    # Drop unnecessary columns, including 'age_cod'
    demo_df.drop(columns=['age_cod', 'age_combined'], inplace=True, errors='ignore')

    # Convert caseversion to numeric type 
    demo_df['caseversion'] = pd.to_numeric(demo_df['caseversion'], errors='coerce')
    # Filter for the latest caseversion for each caseid, and keep the first
    demo_df = demo_df.sort_values('caseversion', ascending=False).drop_duplicates(subset='caseid', keep='first')
    # Filter countries of reporting and occurrences to USA
    demo_df = demo_df[(demo_df['reporter_country'] == 'US') & (demo_df['occr_country'] == 'US')]
    # Drop unwanted columns
    demo_df = demo_df.drop(columns=['mfr_dt', 'init_fda_dt', 'fda_dt', 'auth_num', 'mfr_num', 'mfr_sndr', 'to_mfr', 'lit_ref', 'wt', 'wt_cod', 'age_grp', 'e_sub'])


if reac_file:
    reac_df = pd.concat(reac_file, ignore_index=True)

if outc_file:
    outc_df = pd.concat(outc_file, ignore_index=True)

# Aggregate DataFrames to Avoid Duplicates Before Merging
if drug_file:
    drug_df = pd.concat(drug_file, ignore_index=True)
    
    # Standardize drug names
    drug_df['drugname'] = (
        drug_df['drugname']
        .str.lower()                                     # Convert to lowercase
        .str.replace(r'\b(and|&)\b', ',', regex=True)    # Replace standalone 'and' or '&' with a comma
        .str.replace(r'[\\\/]', ',', regex=True)         # Replace slashes (both / and \) with commas
        .str.replace(r'[^\w\s,]', '', regex=True)        # Remove special characters except commas
        .str.replace(r',\s*', ', ', regex=True)          # Ensure proper spacing after commas
        .str.replace(r'\s+', ' ', regex=True)           # Replace multiple spaces with a single space
        .str.strip()                                    # Strip leading/trailing spaces
    )

    # Aggregate drug data by primaryid (e.g., concatenate drug names and roles)
    drug_df = drug_df.groupby('primaryid').agg({
        'drugname': lambda x: ', '.join(x.dropna().unique()),
        'role_cod': lambda x: ', '.join(x.dropna().unique()),
        'route': lambda x: ', '.join(x.dropna().unique()),
        'dose_vbm': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()

    # Calculate num_drugs and polypharmacy risk
    # Step 1: Split drugname column
    drug_df['drug_list'] = drug_df['drugname'].str.split(',')

    # Step 2: Standardize drug names in list format
    drug_df['drug_list'] = drug_df['drug_list'].apply(lambda x: [drug.strip().lower() for drug in x] if isinstance(x, list) else [])

    # Step 3: Count the number of drugs per primaryid
    drug_df['num_drugs'] = drug_df['drug_list'].apply(len)

    # Step 4: Assign polypharmacy risk levels
    def assign_polypharmacy_risk(num_drugs):
        if num_drugs <= 4:
            return 'Low'
        elif num_drugs <= 8:
            return 'Moderate'
        else:
            return 'High'

    drug_df['polypharmacy_risk'] = drug_df['num_drugs'].apply(assign_polypharmacy_risk)

    # Drop the intermediate 'drug_list' column
    drug_df.drop(columns=['drug_list'], inplace=True)



if reac_file:
    # Aggregate reaction data by primaryid
    reac_df = reac_df.groupby('primaryid').agg({
        'pt': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()

if outc_file:
    # Aggregate outcome data by primaryid
    outc_df = outc_df.groupby('primaryid').agg({
        'outc_cod': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()

if demo_file and drug_file and reac_file and outc_file:
    try:
        # Merge DataFrames
        merged_df = demo_df.merge(drug_df, on='primaryid', how='inner', suffixes=('_demo', '_drug')) \
                           .merge(reac_df, on='primaryid', how='inner', suffixes=('', '_reac')) \
                           .merge(outc_df, on='primaryid', how='left', suffixes=('', '_outc'))

        # Drop duplicate rows if they still exist
        merged_df = merged_df.drop_duplicates(subset='primaryid')

        # Calculate drug-specific risk scores using the processed merged dataset
        all_drugs = merged_df['drugname'].str.split(',').explode().str.strip().str.lower()
        drug_risk_scores = all_drugs.value_counts(normalize=True).reset_index()
        drug_risk_scores.columns = ['drugname', 'risk_score']

        # Map risk scores back to each primaryid in the merged dataset
        risk_score_dict = drug_risk_scores.set_index('drugname')['risk_score'].to_dict()

        # Calculate the maximum risk score for each participant's drug list
        def calculate_risk_score(drug_list, risk_scores):
            if not isinstance(drug_list, list):
                return 0
            scores = [risk_scores.get(drug.strip().lower(), 0) for drug in drug_list]
            return max(scores) if scores else 0

        merged_df['drug_list'] = merged_df['drugname'].str.split(',').apply(
            lambda x: [drug.strip().lower() for drug in x] if isinstance(x, list) else []
        )
        merged_df['drug_risk_score'] = merged_df['drug_list'].apply(lambda x: calculate_risk_score(x, risk_score_dict))

        # Create a binary high-risk feature, using 1% as the threshold
        merged_df['high_risk_drug'] = (merged_df['drug_risk_score'] > 0.010592).astype(int)

        # Drop intermediate 'drug_list' column
        merged_df.drop(columns=['drug_list'], inplace=True)

        # Save the enriched merged DataFrame
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'merged_faers_data.csv')
        merged_df.to_csv(output_path, index=False)

    except KeyError as e:
        print(f"KeyError: {e}. Please ensure all files contain the 'primaryid' column.")
    except pd.errors.MergeError as e:
        print(f"MergeError: {e}. There was an issue with merging due to duplicate columns.")

else:
    print("Not all required files (DEMO21Q1.txt, DRUG21Q1.txt, REAC21Q1.txt, OUTC21Q1.txt) were found in the specified folder.")