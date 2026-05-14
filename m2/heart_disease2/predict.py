
import sys, json, numpy as np, joblib, os

# نفس أعمدة تدريب نموذج القلب (main.py)
FEATURES = ['age','sex','smoke','years','chp','height','weight','fh','active',
            'lifestyle','ihd','hr','dm','bpsys','bpdias','htn','ecgpatt']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = r"C:\xampp\htdocs\SmartClinic\proj\m2\heart_disease2\model.pkl"
SCALER_PATH = r"C:\xampp\htdocs\SmartClinic\proj\m2\heart_disease2\scaler.pkl"

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
        # إن كان التصنيف ثنائي بدالة predict_proba
        proba = model.predict_proba(arr_scaled)[0][1] if hasattr(model, "predict_proba") else float(model.predict(arr_scaled)[0])
        print(f"{proba:.4f}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
