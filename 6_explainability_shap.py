import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

def run_shap_analysis():
    print("Loading prepared dataset...")
    df = pd.read_csv("/Users/aakashsangani/Desktop/CreditAudit/processed_nsso_credit.csv")
    
    # Fill NAs
    df.fillna(0, inplace=True)
    
    # Log scale extreme power-law distributions (wealth)
    power_law_cols = ['Financial_Assets', 'Total_Physical_Assets']
    for col in power_law_cols:
        df[col] = np.log1p(df[col])
        
    categorical_cols = ['State', 'District', 'HH_Type']
    features = ['Is_Rural', 'Is_Minority_Religion', 'Is_Marginalized_Caste', 'Age_Head', 'Edu_Head', 'HH_Size', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets']

    for cat in categorical_cols:
        if cat in df.columns:
            df_cat = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
            df = pd.concat([df, df_cat], axis=1)
            features.extend(df_cat.columns)
            
    X = df[features]
    y = df['Is_Institutional']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Needs a smaller subset for DeepExplainer otherwise it takes hours
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    print(f"Training Model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    
    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = create_model()
    # Fast train
    model.fit(X_train, y_train, epochs=3, batch_size=256, verbose=1)
    
    # SHAP DeepExplainer
    print("Initiating SHAP Deep Explainer Protocol...")
    # Background dataset for SHAP to compute expectation
    background = X_train[np.random.choice(X_train.shape[0], 500, replace=False)]
    e = shap.DeepExplainer(model, background)
    
    # Explaining 500 random sample predictions
    test_sample = X_test[np.random.choice(X_test.shape[0], 500, replace=False)]
    shap_values = e.shap_values(test_sample)
    
    # SHAP returns a list of arrays for Keras models. Take index 0 for output node.
    if isinstance(shap_values, list):
        shap_values_plot = np.squeeze(shap_values[0])
    else:
        shap_values_plot = np.squeeze(shap_values)
        
    print("Generating SHAP Density Plot...")
    
    # Because there are 700+ District dummy features, we only plot the non-dummy features + top 10 dummies
    shap.summary_plot(shap_values_plot, features=test_sample, feature_names=features, max_display=15, show=False)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    # Tweak spacing
    plt.tight_layout()
    plt.savefig('/Users/aakashsangani/Desktop/CreditAudit/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print("SHAP analysis artifact saved successfully!")

if __name__ == "__main__":
    run_shap_analysis()
