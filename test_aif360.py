import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset

df = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'Is_Rural': [1, 0, 1, 0],
    'Is_Institutional': [1, 0, 1, 0],
    'MLT': [1.0, 1.0, 1.0, 1.0]
})

try:
    dataset = BinaryLabelDataset(
        favorable_label=1, unfavorable_label=0, df=df,
        label_names=['Is_Institutional'], protected_attribute_names=['Is_Rural'],
        instance_weights_name='MLT',
        unprivileged_protected_attributes=[[1]],
        privileged_protected_attributes=[[0]]
    )
    print("SUCCESS with [[1]]")
except Exception as e:
    print(f"FAILED with [[1]]: {e}")

try:
    dataset = BinaryLabelDataset(
        favorable_label=1, unfavorable_label=0, df=df,
        label_names=['Is_Institutional'], protected_attribute_names=['Is_Rural'],
        instance_weights_name='MLT',
        unprivileged_protected_attributes=[np.array([1])],
        privileged_protected_attributes=[np.array([0])]
    )
    print("SUCCESS with [np.array([1])]")
except Exception as e:
    print(f"FAILED with [np.array([1])]: {e}")
