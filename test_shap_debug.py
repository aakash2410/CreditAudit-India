import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

df = pd.read_csv('processed_nsso_credit.csv')
for col in ['Financial_Assets', 'Total_Physical_Assets']:
    df[col] = np.log1p(df[col].fillna(0))

categorical_cols = ['State', 'District', 'HH_Type']
features = ['Is_Rural', 'Is_Minority_Religion', 'Is_Marginalized_Caste', 'Age_Head', 'Edu_Head', 'HH_Size', 'Land_Possessed', 'Financial_Assets', 'Total_Physical_Assets']

for cat in categorical_cols:
    if cat in df.columns:
        df_cat = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
        df = pd.concat([df, df_cat], axis=1)
        features.extend(df_cat.columns)

X = df[features]
y = df['Is_Institutional'].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=2, batch_size=256, verbose=0)

background = X_train[np.random.choice(X_train.shape[0], 200, replace=False)]
e = shap.DeepExplainer(model, background)

test_sample = X_test[np.random.choice(X_test.shape[0], 200, replace=False)]
shap_values = e.shap_values(test_sample)

if isinstance(shap_values, list):
    vals = np.abs(shap_values[0]).mean(0)
else:
    vals = np.abs(shap_values).mean(0)

# get top 15 features
feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['col_name', 'feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
print(feature_importance.head(20))
