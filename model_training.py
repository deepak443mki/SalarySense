import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_clean, feature_engineer, split_data

# Load & preprocess data
df = load_and_clean('adult 3.csv')
df = feature_engineer(df)
X_train, X_test, y_train, y_test = split_data(df)

# Feature lists
numeric_features = ['age', 'fnlwgt', 'educational-num', 'hours-per-week']
categorical_features = [
    'workclass','education','marital-status','occupation','relationship',
    'race','gender','native-country','age_bin'
]

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define models to evaluate
model_candidates = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

best_model = None
best_name = None
best_acc = 0.0

# Evaluate each model
for name, clf in model_candidates.items():
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = pipeline
        best_name = name

# Detailed report for best
print(f"\nBest Model: {best_name} with accuracy {best_acc:.4f}\n")
preds_best = best_model.predict(X_test)
print(classification_report(y_test, preds_best))

# Check threshold
if best_acc < 0.90:
    print("Warning: No model achieved 90% accuracy. Consider feature engineering or hyperparameter tuning.")

# Save best model
model_filename = f'salarysense_{best_name.lower()}.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")