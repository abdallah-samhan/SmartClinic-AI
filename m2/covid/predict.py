
import sys
import json
import numpy as np
import joblib
import os

# ملاحظة: هذا الملف هو predict.py القديم الخاص بـ COVID-19
# ويحافظ على key_mapping و feature_order الأصلية

key_mapping = {
    "breathing_problem": "Breathing Problem",
    "fever": "Fever",
    "dry_cough": "Dry Cough",
    "sore_throat": "Sore throat",
    "running_nose": "Running Nose",
    "asthma": "Asthma",
    "chronic_lung_disease": "Chronic Lung Disease",
    "headache": "Headache",
    "heart_disease": "Heart Disease",
    "diabetes": "Diabetes",
    "hyper_tension": "Hyper Tension",
    "fatigue": "Fatigue",
    "gastrointestinal": "Gastrointestinal",
    "abroad_travel": "Abroad travel",
    "contact_with_covid_patient": "Contact with COVID Patient",
    "attended_large_gathering": "Attended Large Gathering",
    "visited_public_exposed_places": "Visited Public Exposed Places",
    "family_working_public_places": "Family working in Public Exposed Places",
    "wearing_masks": "Wearing Masks",
    "sanitization_from_market": "Sanitization from Market"
}

feature_order = list(key_mapping.values())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

def main():
    try:
        # 1) قراءة JSON من stdin
        input_json = sys.stdin.read()
        data = json.loads(input_json)

        # 2) تعبئة المفاتيح الناقصة افتراضياً بـ 0 لتجنب الأخطاء
        for php_key in key_mapping.keys():
            data.setdefault(php_key, 0)

        # 3) تحويل البيانات إلى ترتيب الميزات الأصلي
        mapped = { key_mapping[k]: data[k] for k in key_mapping.keys() }
        values = [mapped[f] for f in feature_order]
        arr = np.array(values, dtype=float).reshape(1, -1)

        # 4) تحميل النموذج والمقياس
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or scaler not found.")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # 5) التحجيم ثم التنبؤ
        arr_scaled = scaler.transform(arr)
        proba = model.predict_proba(arr_scaled)[0][1] if hasattr(model, "predict_proba") \
                else float(model.predict(arr_scaled)[0])

        # 6) طباعة الاحتمال كنص
        print(f"{proba:.4f}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
