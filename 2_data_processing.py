import pandas as pd
import numpy as np
import os
import json

def process_nsso_data():
    base_dir = "/Users/aakashsangani/Desktop/CreditAudit/extracted_data"
    
    print("Loading Demographics (Block 3)...")
    df_demo = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 02 (Block 3) - Demographic and other particulars of household members.csv"), low_memory=False)
    
    print("Loading Household Characteristics (Block 4)...")
    df_hh_chars = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 03 (Block 4) - Household characteristics.csv"), low_memory=False)
    
    print("Loading Financial Assets (Block 11a)...")
    df_assets = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 12 (Block 11a) - Financial assets including receivables (other than shares and related instruments) owned by the household..csv"), low_memory=False)
    
    print("Loading Extensive Physical Capital Assets (Blocks 6-10, 11b)...")
    df_b6 = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 07 (Block 6) - Buildings and other constructions owned by the household.csv"), usecols=['HHID', 'b6q5'], low_memory=False)
    df_b7 = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 08 (Block 7) - Livestock and poultry owned by the household.csv"), usecols=['HHID', 'b7q4'], low_memory=False)
    df_b8 = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 09 (Block 8) - Transport equipment owned by the household.csv"), usecols=['HHID', 'b8q5'], low_memory=False)
    df_b9 = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 10 (Block 9) - Agricultural machinery and implements owned by the household.csv"), usecols=['HHID', 'b9q4'], low_memory=False)
    df_b10 = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 11 (Block 10) - Non-farm business equipment fully owned by the household.csv"), usecols=['HHID', 'b10q3'], low_memory=False)
    df_b11b = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 13 (Block 11b) - Investments in share and related instruments owned by the household in co-operative societies & companies..csv"), usecols=['HHID', 'b11bq6'], low_memory=False)
    
    print("Loading Cash Loans (Block 12)...")
    try:
        df_loans = pd.read_csv(os.path.join(base_dir, "Visit1  Level - 14 (Block 12) - particulars of cash loans payable by the household to institutional, non-institutional agencies as on the date of survey and transactions of loans.csv"), low_memory=False)
    except FileNotFoundError:
        print("Loans file not found, looking for Visit2...")
        df_loans = pd.read_csv(os.path.join(base_dir, "Visit2  Level - 14 (Block 12) - particulars of cash loans payable by the household to institutional, non-institutional agencies as on the date of survey and transactions of loans.csv"))

    # Convert HHID to string to be safe and trim floating point artifacts
    df_demo['HHID'] = df_demo['HHID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_loans['HHID'] = df_loans['HHID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_hh_chars['HHID'] = df_hh_chars['HHID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_assets['HHID'] = df_assets['HHID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # Process physical assets keys
    for df in [df_b6, df_b7, df_b8, df_b9, df_b10, df_b11b]:
        df['HHID'] = df['HHID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
    print("Aggregating Massive Physical Collateral & Net Worth Matrices...")
    
    # Process HH socioeconomics (Block 4)
    df_hh_chars = df_hh_chars[['HHID', 'b4q1', 'b4q2', 'b4q3', 'b4q4', 'b4q5']].copy()
    
    # Process Financial Assets (Block 11a)
    df_assets['Asset_Value'] = pd.to_numeric(df_assets['b11aq3'], errors='coerce').fillna(0)
    hh_assets = df_assets.groupby('HHID')['Asset_Value'].sum().reset_index()
    hh_assets.rename(columns={'Asset_Value': 'Financial_Assets'}, inplace=True)
    
    # Process Total Physical Assets (Aggregating Real Estate, Transport, Farm/Business Equip, Livestock, Shares)
    df_b6['Val'] = pd.to_numeric(df_b6['b6q5'], errors='coerce').fillna(0)
    df_b7['Val'] = pd.to_numeric(df_b7['b7q4'], errors='coerce').fillna(0)
    df_b8['Val'] = pd.to_numeric(df_b8['b8q5'], errors='coerce').fillna(0)
    df_b9['Val'] = pd.to_numeric(df_b9['b9q4'], errors='coerce').fillna(0)
    df_b10['Val'] = pd.to_numeric(df_b10['b10q3'], errors='coerce').fillna(0)
    df_b11b['Val'] = pd.to_numeric(df_b11b['b11bq6'], errors='coerce').fillna(0)

    hh_physical = pd.concat([df_b6, df_b7, df_b8, df_b9, df_b10, df_b11b]).groupby('HHID')['Val'].sum().reset_index()
    hh_physical.rename(columns={'Val': 'Total_Physical_Assets'}, inplace=True)
    
    # We want one row per Household, but Block 3 is at the Person level.
    # We will filter for Head of Household (b3q3 == 1 usually represents self/head) to get household level demography.
    # b3q3 : Relation to head -> 1 = self
    df_hh = df_demo[df_demo['b3q3'] == 1].copy()
    if df_hh.empty:
        # fallback if string or int
        df_hh = df_demo[df_demo['b3q3'] == '1'].copy()
    
    # Extract Multiplier natively present in Block 3 and divide by 100 per NSSO standard
    if 'MLT' in df_hh.columns:
        df_hh['MLT'] = pd.to_numeric(df_hh['MLT'], errors='coerce').fillna(1.0) / 100.0
    else:
        df_hh['MLT'] = 1.0
    
    # Process Demographics: Sector
    # Sector: 1=Rural, 2=Urban
    df_hh['Is_Rural'] = (df_hh['Sector'].astype(str) == '1').astype(int)

    # Process Loans
    # Institutional agencies are codes 01 to 13
    df_loans['b12q5_num'] = pd.to_numeric(df_loans['b12q5'], errors='coerce')
    df_loans['Is_Institutional'] = ((df_loans['b12q5_num'] >= 1) & (df_loans['b12q5_num'] <= 13)).astype(int)
    
    # Aggregate loans to Household level (Does HH have ANY institutional loan?)
    hh_loans = df_loans.groupby('HHID')['Is_Institutional'].max().reset_index()

    print("Merging datasets...")
    # LEFT join demographics with loans to preserve households with NO debt for Stage 1 (Heckman Access Model)
    df_merged = df_hh.merge(hh_loans, on='HHID', how='left')
    df_merged = df_merged.merge(df_hh_chars.drop_duplicates(subset=['HHID']), on='HHID', how='left')
    df_merged = df_merged.merge(hh_assets, on='HHID', how='left')
    df_merged = df_merged.merge(hh_physical, on='HHID', how='left')
    
    # Create Stage 1 outcome variable: In_Credit_Market (1 if they have any loan, 0 if no debt)
    df_merged['In_Credit_Market'] = df_merged['Is_Institutional'].notna().astype(int)
    
    # Fill NaN for households with missing data
    df_merged['Is_Institutional'] = df_merged['Is_Institutional'].fillna(0).astype(int)
    df_merged['Financial_Assets'] = df_merged['Financial_Assets'].fillna(0)
    df_merged['Total_Physical_Assets'] = df_merged['Total_Physical_Assets'].fillna(0)

    # Select features
    features = ['HHID', 'State', 'District', 'Is_Rural', 'Is_Institutional', 'In_Credit_Market', 'MLT']
    
    # Adding age (b3q5), education (b3q6), and sex (b3q4) of head as features
    df_merged['Age_Head'] = pd.to_numeric(df_merged['b3q5'], errors='coerce').fillna(0)
    df_merged['Edu_Head'] = pd.to_numeric(df_merged['b3q6'], errors='coerce').fillna(0)
    # Sex: 1 = Male, 2 = Female. We create a binary Is_Female_Head feature.
    df_merged['Is_Female_Head'] = (pd.to_numeric(df_merged['b3q4'], errors='coerce') == 2).astype(int)
    
    # Adding HH Characteristics
    df_merged['HH_Size'] = pd.to_numeric(df_merged['b4q1'], errors='coerce').fillna(1)
    df_merged['HH_Type'] = pd.to_numeric(df_merged['b4q4'], errors='coerce').fillna(9)
    df_merged['Land_Possessed'] = pd.to_numeric(df_merged['b4q5'], errors='coerce').fillna(0)
    
    # Create Binary Protected Attributes for AIF360 Debiasing
    # Religion: 'Hinduism' = Privileged Majority (0), others = Minority (1)
    df_merged['Is_Minority_Religion'] = (df_merged['b4q2'].astype(str).str.strip().str.lower() != 'hinduism').astype(int)
    
    # Social Group: 9 = General/Upper (Privileged), 1/2/3 = ST/SC/OBC (Marginalized)
    df_merged['Is_Marginalized_Caste'] = (pd.to_numeric(df_merged['b4q3'], errors='coerce') != 9).astype(int)
    
    features.extend(['Is_Female_Head', 'Age_Head', 'Edu_Head', 'HH_Size', 'Is_Minority_Religion', 'Is_Marginalized_Caste', 'HH_Type', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets'])
    
    df_final = df_merged[features].copy()
    
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Rural Household Count (Raw): {df_final['Is_Rural'].sum()}")
    print(f"Households in Credit Market (Raw): {df_final['In_Credit_Market'].sum()}")
    print(f"Institutional Loan Holders (Raw): {df_final['Is_Institutional'].sum()}")
    print(f"Total Scaled Population Represented: {df_final['MLT'].sum():.0f}")
    
    out_path = "/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    process_nsso_data()
