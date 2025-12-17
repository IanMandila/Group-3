import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, accuracy_score
import numpy as np

# Load the cleaned dataset
CLEANED_DATA_PATH = 'adult_cleaned.csv'
df = pd.read_csv(CLEANED_DATA_PATH)

# Separate features (X) and target (y)
X = df.drop('income', axis=1)
y = df['income']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (should be none here)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model 1: Logistic Regression ---
log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

log_reg_pipeline.fit(X_train, y_train)
y_pred_log_reg = log_reg_pipeline.predict(X_test)
y_proba_log_reg = log_reg_pipeline.predict_proba(X_test)[:, 1]

# --- Model 2: Random Forest Classifier ---
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# --- Evaluation Function ---
def evaluate_model(y_test, y_pred, y_proba, model_name):
    """Calculates and prints BI-relevant metrics for a model."""
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate Misclassification Costs (Q3c consideration)
    # Assuming a simple cost model:
    # False Negative (FN): Cost of missing a high-income individual (e.g., lost opportunity for premium product) = 5
    # False Positive (FP): Cost of incorrectly flagging a low-income individual (e.g., wasted marketing effort) = 1
    FN = cm[1, 0]
    FP = cm[0, 1]
    total_misclassification_cost = (FN * 5) + (FP * 1)
    print(f"Total Misclassification Cost (FN*5 + FP*1): {total_misclassification_cost}")

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'Misclassification Cost': total_misclassification_cost
    }

# --- Run Evaluation ---
log_reg_metrics = evaluate_model(y_test, y_pred_log_reg, y_proba_log_reg, "Logistic Regression")
rf_metrics = evaluate_model(y_test, y_pred_rf, y_proba_rf, "Random Forest")

# --- Comparison Table (Q3b) ---
comparison_df = pd.DataFrame([log_reg_metrics, rf_metrics])
comparison_df = comparison_df.set_index('Model')

print("\n--- Model Comparison Table (Q3b) ---")
print(comparison_df.to_markdown(floatfmt=".4f"))

# --- Managerial Interpretation (Q3c) ---
# This part will be integrated into the final report, but we can print a summary here.
print("\n--- Managerial Interpretation Summary (Q3c) ---")
if rf_metrics['AUC'] > log_reg_metrics['AUC']:
    preferred_model = "Random Forest"
    reason = "It has a higher AUC, indicating better overall discriminative power, and a better balance between Precision and Recall."
else:
    preferred_model = "Logistic Regression"
    reason = "It is more interpretable (simpler model) and may be preferred if transparency is paramount, despite slightly lower performance."

print(f"Preferred Model for a real institution: {preferred_model}")
print(f"Reason: {reason}")
print("Note: The final choice depends on the operational trade-offs (e.g., cost of False Positives vs. False Negatives).")
