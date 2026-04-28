import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def build_and_evaluate_stage1():
    print("Loading prepared dataset...")
    df = pd.read_csv("/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv")
    
    # Fill NAs
    df.fillna(0, inplace=True)
    
    # Feature Engineering
    features = ['Is_Female_Head', 'Is_Rural', 'Is_Minority_Religion', 'Is_Marginalized_Caste', 'Age_Head', 'Edu_Head', 'HH_Size', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets']
    
    # Log scale extreme power-law distributions (wealth)
    power_law_cols = ['Financial_Assets', 'Total_Physical_Assets']
    for col in power_law_cols:
        df[col] = np.log1p(df[col])
        
    # One-hot encode categoricals
    categorical_cols = ['State', 'District', 'HH_Type']
    for cat in categorical_cols:
        if cat in df.columns:
            df_cat = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
            df = pd.concat([df, df_cat], axis=1)
            features.extend(df_cat.columns)
            
    X = df[features]
    weights = df['MLT']
    y = df['In_Credit_Market']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DF to track the index
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X_scaled, y, weights, test_size=0.2, random_state=42)
    
    print("Building Stage 1 (Market Access) TF Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training Stage 1 Model...")
    model.fit(X_train, y_train, sample_weight=w_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    # --- FAIRNESS AUDIT with AIF360 ---
    print("\nAuditing Stage 1 Model across Multi-Dimensional Protected Attributes...")
    
    # Reconstruct datasets
    df_test_raw = df.loc[X_test.index].copy()
    df_test_raw['In_Credit_Market'] = y_test
    df_test_raw['MLT'] = w_test
    
    df_pred_raw = df_test_raw.copy()
    df_pred_raw['In_Credit_Market'] = y_pred

    protected_attributes = {
        'Is_Rural': ('RURAL VS URBAN', 1, 0),
        'Is_Minority_Religion': ('MINORITY VS MAJORITY RELIGION', 1, 0),
        'Is_Marginalized_Caste': ('SC/ST/OBC VS GENERAL CASTE', 1, 0),
        'Is_Female_Head': ('FEMALE VS MALE HEAD', 1, 0)
    }

    for attr, (desc, unpriv_val, priv_val) in protected_attributes.items():
        dataset_orig = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_test_raw,
            label_names=['In_Credit_Market'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )
        
        dataset_pred = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_pred_raw,
            label_names=['In_Credit_Market'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )

        try:
            metric = ClassificationMetric(dataset_orig, dataset_pred, 
                                          unprivileged_groups=[{attr: unpriv_val}], 
                                          privileged_groups=[{attr: priv_val}])
            
            print("\n" + "="*50)
            print(f"STAGE 1 (ACCESS) MODEL FAIRNESS: {desc}")
            print("="*50)
            print(f"Accuracy:                  {(y_pred == y_test).mean():.4f}")
            print(f"Disparate Impact:          {metric.disparate_impact():.4f} (Ideal: 1.0)")
            print(f"Equal Opportunity Diff:    {metric.equal_opportunity_difference():.4f} (Ideal: 0.0)")
            print(f"Statistical Parity Diff:   {metric.statistical_parity_difference():.4f} (Ideal: 0.0)")
            print("="*50)
        except Exception as e:
            print(f"Could not compute metrics for {desc}: {e}")

if __name__ == "__main__":
    build_and_evaluate_stage1()
