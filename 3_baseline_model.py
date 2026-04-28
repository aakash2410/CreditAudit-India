import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def build_and_evaluate_baseline():
    print("Loading prepared dataset...")
    df = pd.read_csv("/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv")
    
    # Fill NAs
    df.fillna(0, inplace=True)
    
    # Filter for Stage 2: Only those in the credit market
    print("Filtering for Stage 2 (Allocation Model): Only households with debt...")
    df = df[df['In_Credit_Market'] == 1].copy()
    
    # Base Features
    base_features = [
        'Is_Female_Head', 'Is_Rural', 'Is_Minority_Religion', 'Is_Marginalized_Caste', 
        'Age_Head', 'Edu_Head', 'HH_Size', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets',
        'Age_Group_Young', 'Age_Group_Senior', 'Per_Capita_Physical', 'Per_Capita_Financial',
        'Has_Zero_Land', 'Has_Zero_Financial'
    ]
    
    # Log scale extreme power-law distributions (wealth)
    power_law_cols = ['Financial_Assets', 'Total_Physical_Assets', 'Per_Capita_Physical', 'Per_Capita_Financial']
    for col in power_law_cols:
        df[col] = np.log1p(df[col])
        
    # One-hot encode categoricals (except District)
    categorical_cols = ['State', 'HH_Type']
    for cat in categorical_cols:
        if cat in df.columns:
            df_cat = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
            df = pd.concat([df, df_cat], axis=1)
            base_features.extend(df_cat.columns)
            
    # Include District for now, we will target encode it after splitting
    features = base_features + ['District']
            
    X = df[features]
    weights = df['MLT']
    y = df['Is_Institutional']
    
    # Strict Train-Test Split BEFORE Target Encoding & Scaling to prevent data leakage
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)
    
    # --- TARGET ENCODING ---
    print("Target Encoding 'District' to prevent sparse matrix dilution...")
    # Calculate means on train only
    train_districts = pd.DataFrame({'District': X_train['District'], 'Target': y_train})
    district_means = train_districts.groupby('District')['Target'].mean()
    
    # Map to Train and Test. Fill unseen districts in test with global mean of train
    global_mean = y_train.mean()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['District_Target_Enc'] = X_train['District'].map(district_means).fillna(global_mean)
    X_test['District_Target_Enc'] = X_test['District'].map(district_means).fillna(global_mean)
    
    # Drop original District string column and add the encoded column to features list
    X_train.drop('District', axis=1, inplace=True)
    X_test.drop('District', axis=1, inplace=True)
    final_features = [f for f in features if f != 'District'] + ['District_Target_Enc']
    
    # --- SCALING ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DF to track the index
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=final_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=final_features, index=X_test.index)
    
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    
    print("Building Expert-Crafted Baseline TF Model...")
    from tensorflow.keras.regularizers import l2
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Advanced Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    
    print("Training Optimized Baseline Model...")
    model.fit(
        X_train_scaled, y_train, 
        sample_weight=w_train, 
        epochs=15, 
        batch_size=128, 
        validation_split=0.2, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    # --- FAIRNESS AUDIT with AIF360 ---
    print("\nAuditing Optimized Baseline Model across Multi-Dimensional Protected Attributes...")
    
    # Reconstruct datasets for AIF360
    df_test_raw = df.loc[X_test_scaled.index].copy()
    df_test_raw['Is_Institutional'] = y_test
    df_test_raw['MLT'] = w_test
    
    df_pred_raw = df_test_raw.copy()
    df_pred_raw['Is_Institutional'] = y_pred

    protected_attributes = {
        'Is_Rural': ('RURAL VS URBAN', 1, 0),
        'Is_Minority_Religion': ('MINORITY VS MAJORITY RELIGION', 1, 0),
        'Is_Marginalized_Caste': ('SC/ST/OBC VS GENERAL CASTE', 1, 0),
        'Is_Female_Head': ('FEMALE VS MALE HEAD', 1, 0)
    }

    for attr, (desc, unpriv_val, priv_val) in protected_attributes.items():
        dataset_orig = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_test_raw,
            label_names=['Is_Institutional'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )
        
        dataset_pred = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_pred_raw,
            label_names=['Is_Institutional'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )

        try:
            metric = ClassificationMetric(dataset_orig, dataset_pred, 
                                          unprivileged_groups=[{attr: unpriv_val}], 
                                          privileged_groups=[{attr: priv_val}])
            
            print("\n" + "="*50)
            print(f"OPTIMIZED BASELINE FAIRNESS: {desc}")
            print("="*50)
            print(f"Accuracy:                  {(y_pred == y_test).mean():.4f}")
            print(f"Disparate Impact:          {metric.disparate_impact():.4f} (Ideal: 1.0)")
            print(f"Equal Opportunity Diff:    {metric.equal_opportunity_difference():.4f} (Ideal: 0.0)")
            print(f"Statistical Parity Diff:   {metric.statistical_parity_difference():.4f} (Ideal: 0.0)")
            print("="*50)
        except Exception as e:
            print(f"Error calculating metrics for {desc}: {e}")

if __name__ == "__main__":
    build_and_evaluate_baseline()
