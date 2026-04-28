import pandas as pd
df_demo = pd.read_csv('extracted_data/Visit1  Level - 02 (Block 3) - Demographic and other particulars of household members.csv', low_memory=False)
df_hh_chars = pd.read_csv('extracted_data/Visit1  Level - 03 (Block 4) - Household characteristics.csv', low_memory=False)
df_loans = pd.read_csv('extracted_data/Visit1  Level - 14 (Block 12) - particulars of cash loans payable by the household to institutional, non-institutional agencies as on the date of survey and transactions of loans.csv', low_memory=False)

df_demo['HHID'] = df_demo['HHID'].astype(str)
df_hh_chars['HHID'] = df_hh_chars['HHID'].astype(str)
df_loans['HHID'] = df_loans['HHID'].astype(str)

df_hh = df_demo[df_demo['b3q4'] == '1'].copy()
df_hh_chars = df_hh_chars[['HHID', 'b4q1', 'b4q2', 'b4q3', 'b4q4', 'b4q5']].copy()

hh_loans = df_loans[['HHID']].drop_duplicates()

df_merged = df_hh.merge(hh_loans, on='HHID', how='inner')
df_merged = df_merged.merge(df_hh_chars.drop_duplicates(subset=['HHID']), on='HHID', how='left')

print("b4q2 value counts in merged df:")
print(df_merged['b4q2'].value_counts(dropna=False))
