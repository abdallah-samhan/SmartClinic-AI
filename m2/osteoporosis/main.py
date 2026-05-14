
# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
import joblib
import json

# 1) تحميل البيانات
CSV_PATH = 'osteoporosis.csv'
df = pd.read_csv(CSV_PATH)

TARGET = 'Osteoporosis'
ID_COLS = ['Id']

# أعمدة الميزات مبنية على رأس ملفك
FEATURES = [c for c in df.columns if c not in ID_COLS + [TARGET]]

X = df[FEATURES]
y = df[TARGET]

# 2) تقسيم البيانات (طبقي)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3) المعالجة المسبقة
numeric_feats = [c for c in FEATURES if pd.api.types.is_numeric_dtype(df[c])]
categorical_feats = [c for c in FEATURES if c not in numeric_feats]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
])

# 4) نموذج التصنيف
clf = RandomForestClassifier(
    n_estimators=700,
    max_depth=16,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', clf)
])

# 5) تدريب
pipe.fit(X_train, y_train)

# 6) تقييم
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.50).astype(int)  # عتبة افتراضية 0.5

metrics = {
    'roc_auc': float(roc_auc_score(y_test, y_proba)),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}

print("\n=== نتائج الاختبار ===")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"Confusion Matrix: {metrics['confusion_matrix']}")

# 7) حفظ البايبلاين (النموذج + المعالجة)
MODEL_PATH = 'osteoporosis_model.pkl'
joblib.dump(pipe, MODEL_PATH)

# حفظ ميتا (يمكن تعديل العتبة لاحقًا حسب هدفك)
META_PATH = 'osteoporosis_meta.json'
meta = {
    'features': FEATURES,
    'numeric_feats': numeric_feats,
    'categorical_feats': categorical_feats,
    'threshold': 0.50,  # عتبة افتراضية
    'metrics': metrics
}
with open(META_PATH, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nSaved model → {MODEL_PATH}")
print(f"Saved meta  → {META_PATH}")
