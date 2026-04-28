import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.metrics import ClassificationMetric

# AIF360 Adversarial Debiasing requires tf.compat.v1 without Eager Execution
tf.disable_eager_execution()

def build_adversarial_debiasing_model():
    print("Loading prepared dataset...")
    df = pd.read_csv("/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv")
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
            
    # Strict Train-Test Split BEFORE Target Encoding & Scaling
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # --- TARGET ENCODING ---
    print("Target Encoding 'District'...")
    train_districts = df_train.groupby('District')['Is_Institutional'].mean()
    global_mean = df_train['Is_Institutional'].mean()
    
    df_train['District_Target_Enc'] = df_train['District'].map(train_districts).fillna(global_mean)
    df_test['District_Target_Enc'] = df_test['District'].map(train_districts).fillna(global_mean)
    
    features = base_features + ['District_Target_Enc']
    
    # --- SCALING ---
    scale_cols = ['Age_Head', 'Edu_Head', 'HH_Size', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets', 'Per_Capita_Physical', 'Per_Capita_Financial']
    scaler = StandardScaler()
    df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
    df_test[scale_cols] = scaler.transform(df_test[scale_cols])
    
    protected_attributes = {
        'Is_Rural': ('RURAL VS URBAN', 1, 0),
        'Is_Minority_Religion': ('MINORITY VS MAJORITY RELIGION', 1, 0),
        'Is_Marginalized_Caste': ('SC/ST/OBC VS GENERAL CASTE', 1, 0),
        'Is_Female_Head': ('FEMALE VS MALE HEAD', 1, 0)
    }

    print("\nExecuting AIF360 Adversarial Debiasing Models Across All Dimensions...")
    
    for attr, (desc, unpriv_val, priv_val) in protected_attributes.items():
        # Create Train Dataset
        dataset_train = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_train[features + ['Is_Institutional', 'MLT']],
            label_names=['Is_Institutional'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )
        
        # Create Test Dataset
        dataset_test = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=df_test[features + ['Is_Institutional', 'MLT']],
            label_names=['Is_Institutional'], protected_attribute_names=[attr],
            instance_weights_name='MLT',
            unprivileged_protected_attributes=[[unpriv_val]],
            privileged_protected_attributes=[[priv_val]]
        )
        
        sess = tf.Session()
        debiased_model = AdversarialDebiasing(
            privileged_groups=[{attr: priv_val}],
            unprivileged_groups=[{attr: unpriv_val}],
            scope_name=f'debiased_classifier_{attr}',
            debias=True, sess=sess, num_epochs=10, batch_size=256,
            classifier_num_hidden_units=256
        )
        
        debiased_model.fit(dataset_train)
        dataset_debiasing_test = debiased_model.predict(dataset_test)
        
        metric = ClassificationMetric(dataset_test, dataset_debiasing_test, 
                                      unprivileged_groups=[{attr: unpriv_val}], 
                                      privileged_groups=[{attr: priv_val}])

        print("\n" + "="*50)
        print(f"ADVERSARIAL FAIRNESS: {desc}")
        print("="*50)
        print(f"Accuracy:                  {(dataset_debiasing_test.labels == dataset_test.labels).mean():.4f}")
        print(f"Disparate Impact:          {metric.disparate_impact():.4f} (Ideal: 1.0)")
        print(f"Equal Opportunity Diff:    {metric.equal_opportunity_difference():.4f} (Ideal: 0.0)")
        print(f"Statistical Parity Diff:   {metric.statistical_parity_difference():.4f} (Ideal: 0.0)")
        print("="*50)
        sess.close()
        tf.reset_default_graph()

if __name__ == "__main__":
    build_adversarial_debiasing_model()
