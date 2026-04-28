# Re-Engineering Credit Equity: 77th Round NSSO Audit 

This repository contains an advanced, econometrically-grounded Artificial Intelligence audit of India's credit landscape using the NSSO 77th Round Socio-Economic survey data. 

By applying a **Two-Stage (Heckman-style) Deep Learning Framework** coupled with **Adversarial Debiasing** and **SHAP Explainability**, this project moves beyond simplistic bias metrics to rigorously evaluate structural financial exclusion versus institutional capital allocation.

---

## 📄 Concept Note

**The Problem:** Traditional algorithmic fairness audits often fall into the trap of **Survivorship Bias**. If we only analyze households that successfully secured a bank loan, or even those who currently hold any debt, we ignore the populations suffering from extreme financial redlining—those who are completely excluded from the market or discouraged from even applying. 

**The Duality of Zero-Debt:** In developmental economics, having "zero debt" is not always a sign of wealth or self-sufficiency. For marginalized demographics, it is frequently a sign of absolute credit rationing. 

**The Solution:** To make this audit mathematically robust, we deployed a **Two-Stage Selection Model**:
1.  **Stage 1 (Market Access):** Before predicting *who* gets a bank loan, the AI first predicts the structural probability of a household participating in the credit market at all. 
2.  **Stage 2 (Allocation Disparities):** Given that a household is actively in the credit market (relies on debt), the AI predicts whether they secure safe, formal capital (Institutional Banks/Co-ops) or are forced into predatory lending (Informal Moneylenders).

**The Defense:** This audit explicitly acknowledges that it suffers from omitted variables (like CIBIL scores or true unobserved cash flows) and relies on self-reported survey data rather than RBI origination logs. Therefore, this is not a causal proof of intentional redlining by individual loan officers. Rather, it is a **sociological audit of Credit Reliance Outcomes**. It proves that even when an AI accounts for all measurable physical collateral a household owns, the algorithm still relies heavily on geographic, caste, and gender proxies to predict who ends up with safe formal capital versus predatory informal debt.

---

## 📊 Key Findings

By analyzing over 116,000 instances scaled to represent 260 million Indian households, the Two-Stage model revealed profound insights into how bias operates in two distinct phases:

### Phase 1: Structural Exclusion (Market Access)
The Stage 1 access model reveals extreme disparities in who holds debt in the first place:
*   **Gender Exclusion:** Female-headed households have roughly **59%** of the probability of entering the credit market compared to male-headed households (Disparate Impact: 0.59). This indicates a massive structural barrier to credit access.
*   **Subsistence Reliance:** Conversely, Rural households and Marginalized Castes (SC/ST/OBC) have a *significantly higher* incidence of holding debt than Urban/General caste households (DI: 2.92 and 1.31 respectively). This reflects a heavy reliance on debt for subsistence and agriculture, rather than financial inclusion.

### Phase 2: Capital Allocation Disparities
Once households are in the debt market, who gets the safe institutional loans?
*   **The Baseline Optimization:** By engineering advanced socio-economic features (Per-Capita wealth, Zero-Value Indicators) and applying Target Encoding to over 700 districts, the Baseline AI achieved an optimized predictive accuracy of **72.81%**. We found that Rural, Minority Religion, and Marginalized Caste households all face institutional allocation disparities (DI < 1.0).
*   **Adversarial Correction:** By explicitly penalizing the Predictor network if an Adversary network could guess a household's demographic from its latent weights (Latent Disentanglement), we flattened the proxy-weaponization across all four dimensions. The Adversarial model successfully erased the bias while maintaining a highly robust **71.77%** accuracy, proving we can mathematically force fair allocation without sacrificing predictive power.

---

## 🛠 Methodology Pipeline

The end-to-end framework bypasses conventional limitations to isolate pure demographic vectors through rigorous programmatic pipeline staging:

### 1. Proprietary Extraction (`nesstar-reader`, macOS Bypass)
The NSSO `Round77sch18pt2Data` depends natively on `.Nesstar` structures bound to 32-bit Windows ecosystems. By leveraging the updated `nesstar-reader` Python library on a generic `venv`, we bypassed metadata assertions and brute-force mapped the target memory footprint for the `.csv` generation native to bash environments. 

### 2. Multi-Dimensional Join Processing (`2_data_processing.py`)
Using explicit schemas, we unified Demographics (`Block 3`), Household Structural Features (`Block 4`), Liabilities/Loans (`Block 12`), and Financial Assets (`Block 11a`).
*   **The Scaler**: Embedded `MLT / 100` combination weights representing 260 Million real Indian Households.
*   **Advanced Feature Engineering**: Engineered `Per_Capita_Assets` (Asset / HH Size) and Boolean `Has_Zero_Wealth` indicators to help the deep learning models properly scale extreme poverty boundaries.
*   **Collateral Matrices**: Aggregated `Total_Physical_Assets` across 6 distinct capital matrices in the NSSO database (Real Estate, Livestock, Vehicles, Business Machinery, Shares).
*   **Two-Stage Framing**: Executed a `left` join to preserve zero-debt households, creating the `In_Credit_Market` flag for Stage 1.

### 3. Stage 1: The Access Model (`stage1_access_model.py`)
A multi-layered Keras Sequential model designed to predict `In_Credit_Market`. This model captures the pure systemic exclusion constraints against Female-headed households and Religious Minorities.

### 4. Stage 2: Deep Networks & Adversarial Debiasing (`3_baseline_model.py` & `4_adversarial_model.py`)
*   **Target Encoding**: Replaced sparse 700-column one-hot district matrices with dense Target Encodings (computed strictly on the train split to prevent leakage).
*   **Expert-Crafted Baseline**: A deeply parameterized 256-node Dense Network featuring `BatchNormalization` and `L2 Ridge Regularization` to prevent overfitting on outliers, achieving ~72.8% allocation accuracy.
*   **Adversarial Setup**: Executed multi-threaded independent Adversarial Debiasing networks via `IBM AIF360`, aggressively punishing classification gradients whenever `Is_Rural`, `Is_Minority`, `Is_Marginalized_Caste`, or `Is_Female_Head` became structurally derivable from the prediction latent weights.

---

## ⚖️ Methodological Limitations & Disclaimers

To ensure rigorous interpretation of these findings, please note the following econometric limitations:

1.  **Data Source Reality:** This analysis relies on the NSSO 77th Round, which is self-reported socio-economic household data. It is **not** RBI origination or underwriting log data. We are auditing the ultimate socio-economic reality of "Credit Reliance Outcomes", not internal banking decisions.
2.  **Collider Bias Risk:** A true econometric Heckman Selection Model utilizes an Inverse Mills Ratio to connect Stage 1 to Stage 2. Our Deep Learning approach filters Stage 2 conditionally. While conceptually accurate for modeling real-world outcomes, it is subject to collider bias constraints.
3.  **SHAP represents Algorithmic Compliance, not Causal Inference:** The SHAP explainability analysis used to monitor proxies proves that our *model* is behaving fairly (ensuring the algorithm doesn't mathematically weaponize geography or assets). It does **not** causally prove the presence or absence of systemic bigotry in real-world bank branches.
