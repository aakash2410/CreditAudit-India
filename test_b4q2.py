import pandas as pd
df_demo = pd.read_csv('extracted_data/Visit1  Level - 02 (Block 3) - Demographic and other particulars of household members.csv', low_memory=False)
df_hh_chars = pd.read_csv('extracted_data/Visit1  Level - 03 (Block 4) - Household characteristics.csv', low_memory=False)

df_hh = df_demo[df_demo['b3q4'] == 1].copy()

# The bug is here! Is b4q2 matched on HHID properly?
# Notice how we merge...
df_merged = df_hh.merge(df_hh_chars, on='HHID', how='left')

print("b4q2 values:")
print(df_merged['b4q2'].head())
print("Null count:", df_merged['b4q2'].isnull().sum())
