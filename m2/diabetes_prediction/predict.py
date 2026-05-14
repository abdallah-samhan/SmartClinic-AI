
import sys, json, numpy as np, joblib, os

# مثال شائع — عدّل حسب نموذجك الفعلي إن لزم
FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'blood_glucose_level']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

def main():
    try:
        data = json.loads(sys.stdin.read())
        values = [float(data.get(k, 0)) for k in FEATURES]
        arr = np.array(values, dtype=float).reshape(1, -1)

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or scaler not found.")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        arr_scaled = scaler.transform(arr)
        proba = model.predict_proba(arr_scaled)[0][1] if hasattr(model, "predict_proba") else float(model.predict(arr_scaled)[0])
        print(f"{proba:.4f}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
