
# predict.py
import sys
import json
import pandas as pd
import joblib
import os

MODEL_PATH = 'osteoporosis_model.pkl'
META_PATH  = 'osteoporosis_meta.json'

def main():
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            raise ValueError("لم يتم تمرير JSON عبر stdin.")
        data = json.loads(raw)

        # سجل واحد أو قائمة سجلات
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("المدخل يجب أن يكون كائن JSON أو قائمة كائنات.")

        # تحميل الميتا والبايبلاين
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        pipe = joblib.load(MODEL_PATH)

        threshold = 0.50
        features = None
        if os.path.exists(META_PATH):
            meta = json.load(open(META_PATH, 'r', encoding='utf-8'))
            threshold = float(meta.get('threshold', 0.50))
            features = meta.get('features', None)

        df = pd.DataFrame(records)

        # في حال لديك قائمة ميزات محددة من الميتا، تأكّد من وجودها كلها
        if features is not None:
            for c in features:
                if c not in df.columns:
                    df[c] = None  # اسمح بأعمدة ناقصة؛ OneHot سيهتم بها
            X = df[features]
        else:
            X = df  # fallback—إن لم تتوفر قائمة الميزات

        # تنبؤ: البايبلاين pipe يحوي المعالجة + النموذج
        proba = pipe.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        out = []
        for p, z in zip(proba, pred):
            out.append({'probability': float(p), 'prediction': int(z)})

        print(json.dumps(out, ensure_ascii=False))

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
