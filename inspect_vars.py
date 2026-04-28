import pandas as pd
df1 = pd.read_csv('extracted_data/Visit1  Level - 03 (Block 4) - Household characteristics.csv', low_memory=False)
print("Block 4 b4q2 first 5:")
print(df1['b4q2'].head())
print("b4q2 unique:", df1['b4q2'].unique()[:5])
