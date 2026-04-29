import pandas as pd
from aif360.datasets import BinaryLabelDataset

df = pd.read_csv("/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv")
df.fillna(0, inplace=True)
df = df[df['In_Credit_Market'] == 1].copy()

# Add dummy Is_Institutional and MLT
df['Is_Institutional'] = 1
df['MLT'] = 1.0

attr = 'Is_Rural'
unpriv_val = 1
priv_val = 0

try:
    dataset_orig = BinaryLabelDataset(
        favorable_label=1, unfavorable_label=0, df=df,
        label_names=['Is_Institutional'], protected_attribute_names=[attr],
        instance_weights_name='MLT',
        unprivileged_protected_attributes=[[unpriv_val]],
        privileged_protected_attributes=[[priv_val]]
    )
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
