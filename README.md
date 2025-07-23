# Telco Customer Churn Prediction: A Data-Driven Approach to Retention

## Business Understanding

1. In the telecom industry, customer churn isn’t just a metric—it’s a financial tsunami.
2. A 20% churn rate in a 1M-subscriber base could cost $120M annually (Tridens, 2025).
3. Acquiring new customers costs 6–7x more than retaining existing ones.
4. Even a 0.01% reduction in churn can save $6.6M/year for a major carrier (UST, 2025).
5. 5. AI-driven churn prediction works: T-Mobile slashed churn by 20% using AI-powered retention strategies

## Project Goal
Predict which customers are likely to churn using machine learning, enabling targeted retention efforts to:
- Save revenue by preventing high-risk churn.
- Optimize marketing spend by focusing on customers who need intervention.
- Improve customer lifetime value through data-driven decisions.

## Key Insights & Impact

1. **Model Performance & Cost Savings**
- **Best Model: LightGBM (LGBM)** with **F2-score optimization** (recall-focused) achieved:
    - **95% recall (only 14 high-risk churners missed)**.
    - **$265K net savings** by retaining **$280K revenue** at a **$14.8K treatment cost (ROI: 17.98x)**.
- **Rule-based methods failed**: **Apriori (76% accuracy) missed 220 churners**, **costing $253K—proving accuracy alone is misleading**.

2. **High-Risk Customer Segments**
- **47.6% of new customers (Tenure <12mo) churn** → Critical for onboarding improvements.
- **Fiber optic + No TechSupport customers churn at 41.4%** → Bundling add-ons reduces risk.
- **Month-to-Month contracts churn 2.5x more than long-term contracts**.

3. **Actionable Business Strategies**
- **Executive Leadership:** Fund **First-Year Retention Programs** to reduce early churn.
- **Product Team:** Bundle **TechSupport + DeviceProtection** for high-risk segments.
- **Marketing:** Target **HighSpender_MonthToMonth** customers with **personalized discounts**.
- **Customer Service:** Proactively engage **Very High Risk customers** (34.5% of churners).

**Evaluation Metric: F2-Score**
- Prioritizes **recall over precision** (missing a churner costs **$1,150**, false alarm costs **$50**).
- **LGBM** at **0.208** threshold optimized this trade-off.

## Model Tested

1. **LightGBM:**
- **Low total loss:** $33,600
- **High F2 Score:** 0.7503
- **Good Accuracy:** 0.63
- **Recall and Precision:** 95% and 41%
- **RUC Curve:** 0.833 with minimum overfitting (train and test data gap: 0.04)

**Recommendations for Scalability**
1. **Hyperparameter Tuning:** Optimize **scale_pos_weight** to handle class imbalance.
2. **Feature Engineering:** Create interaction terms (e.g., **Tenure × MonthlyCharges**).
3. **Cost-Sensitive Learning:** Directly minimize $ loss (not just F2-score).
4. **Real-Time Deployment:** Integrate model with CRM to trigger automated retention offers.

## Why This Project Stands Out
1. **Real-world impact:** Directly ties ML metrics to revenue saved.
2. **Strategic segmentation:** Identifies who to save, how, and why.
3. **Scalable framework:** Adaptable to other industries (e.g., SaaS, banking).

Deploy link: https://telcochurnpredictionml.streamlit.app/
  
