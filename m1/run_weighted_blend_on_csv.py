# -*- coding: utf-8 -*-
"""
ملف: per_class_feature_importance.py
الغرض: توليد ملف Excel يحتوي على أهمية الأعراض لكل قسم (Per-Class Feature Importance)
من النماذج المحفوظة في dept_models.pkl باستخدام منهجية الأهمية بالاستبدال (Permutation Importance)
لكل فئة (قسم) وبالنسبة للمزج الوزني (best_weights) وأيضًا لكل نموذج منفرد.

الاستخدام:
python per_class_feature_importance.py --pkl m1/dept_models.pkl --csv grouped_by_department_merged.csv --target Department
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score

# ===== مجموعات الميزات =====
GROUPS = {
    "respiratory_idx": ['cough','wheez','breath','sputum','chest','dyspnea','asthma','pneumonia','bronch','apnea','congestion'],
    "cardiac_idx"    : ['heart','cardio','coronary','myocard','pulse','blood pressure','hypertens','tachy','cyanosis'],
    "neurology_idx"  : ['seiz','paralys','tremor','numb','weak','headache','migraine','vision','optic','speech','walking','confusion','dizziness','balance','myelitis','neuritis'],
    "ent_idx"        : ['ear','throat','nose','sinus','tonsil','hoarseness','post-nasal'],
    "derm_idx"       : ['rash','itch','skin','psoriasis','eczema','acne','scaly','thickening','redness','lesion','pigment'],
    "gi_idx"         : ['abdominal','diarr','vomit','liver','hepat','pancrea','bowel','stool','colitis','crohn','append','spleen','ascites','jaundice','hepatitis'],
    "uro_idx"        : ['urine','urinary','bladder','kidney','renal','testic','prostat','pelvic','neph'],
    "heme_idx"       : ['bleed','anemia','bruis','clot','lymph','spleen','platelet'],
    "endocrine_idx"  : ['thyroid','hypo','hyper','diabet','glycem','hormone'],
    "immune_idx"     : ['auto','lupus','raynaud'],
}
DERIVED_COLS = ["count_positive", "pct_positive"] + list(GROUPS.keys())

# ===== دوال مساعدة =====
def add_group_index(X: pd.DataFrame, features: List[str], name: str, keywords: List[str]):
    cols = np.array(features)
    mask = np.zeros(len(cols), dtype=bool)
    for kw in keywords:
        mask |= np.array([kw.lower() in c.lower() for c in cols])
    X[name] = X.loc[:, cols[mask]].sum(axis=1)
    return X

def proba_in_le_order(model, X: pd.DataFrame, n_classes: int, le):
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    est_classes = getattr(model, "classes_", None)
    if est_classes is None:
        aligned[:, :proba.shape[1]] = proba
        return aligned
    est_classes = np.array(est_classes)
    if est_classes.dtype.kind in {"U", "S", "O"}:
        est_idx = le.transform(est_classes)
    else:
        est_idx = est_classes.astype(int)
    aligned[:, est_idx] = proba
    return aligned

def blend_proba(X: pd.DataFrame, models_for_blend: Dict[str, object], weights: Dict[str, float], n_classes: int, le):
    proba_sum = None
    total_w = 0.0
    for key, mdl in models_for_blend.items():
        w = float(weights.get(key, 0.0))
        if mdl is None or w <= 0:
            continue
        proba = proba_in_le_order(mdl, X, n_classes, le)
        proba_sum = proba if proba_sum is None else (proba_sum + w * proba)
        total_w += w
    if proba_sum is None:
        return np.zeros((X.shape[0], n_classes), dtype=float)
    return proba_sum / max(total_w, 1e-9)

def per_class_f1(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_classes: int) -> np.ndarray:
    f1s = []
    for k in range(n_classes):
        y_true_k = (y_true_idx == k).astype(int)
        y_pred_k = (y_pred_idx == k).astype(int)
        try:
            f1_k = f1_score(y_true_k, y_pred_k)
        except Exception:
            f1_k = 0.0
        f1s.append(float(f1_k))
    return np.array(f1s, dtype=float)

def build_features_like_training(df: pd.DataFrame, feature_cols_training: List[str], target_col: str, ignore_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    base_features = [c for c in df.columns if c not in ignore_cols + [target_col]]
    X = df[base_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X["count_positive"] = X.sum(axis=1)
    X["pct_positive"] = X["count_positive"] / (len(base_features) if len(base_features) else 1)
    for gname, kws in GROUPS.items():
        X = add_group_index(X, base_features, gname, kws)
    missing = [c for c in feature_cols_training if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X.loc[:, feature_cols_training]
    y = df[target_col].astype(str)
    return X, y, base_features

def compute_permutation_importance_per_class(
    X: pd.DataFrame,
    y_idx: np.ndarray,
    n_classes: int,
    proba_fn,
    random_state: int = 42,
    n_repeats: int = 3,
    features_to_eval: List[str] = None,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    features = features_to_eval or list(X.columns)
    proba_base = proba_fn(X)
    y_pred_idx_base = np.argmax(proba_base, axis=1)
    f1_base_per_class = per_class_f1(y_idx, y_pred_idx_base, n_classes)
    rows = []
    print("🔹 حساب الأهمية لكل ميزة...")
    for i, feat in enumerate(features, start=1):
        if i % 10 == 0 or i == len(features):
            print(f"   ميزة {i}/{len(features)}: {feat}")
        drops = []
        for r in range(n_repeats):
            X_perm = X.copy()
            vals = X_perm[feat].values.copy()
            rng.shuffle(vals)
            X_perm[feat] = vals
            proba_perm = proba_fn(X_perm)
            y_pred_idx_perm = np.argmax(proba_perm, axis=1)
            f1_perm_per_class = per_class_f1(y_idx, y_pred_idx_perm, n_classes)
            drop = np.maximum(0.0, f1_base_per_class - f1_perm_per_class)
            drops.append(drop)
        mean_drop = np.mean(np.vstack(drops), axis=0)
        for k in range(n_classes):
            rows.append({"feature": feat, "class_idx": k, "importance": float(mean_drop[k])})
    df_imp = pd.DataFrame(rows)
    return df_imp

# ===== البرنامج الرئيسي =====
def main():
    parser = argparse.ArgumentParser(description="توليد أهمية الأعراض لكل قسم من النماذج المحفوظة")
    parser.add_argument("--pkl", type=str, default="m1/dept_models.pkl", help="مسار ملف الحزمة dept_models.pkl")
    parser.add_argument("--csv", type=str, required=True, help="مسار CSV الذي يحوي أمثلة مع عمود القسم")
    parser.add_argument("--target", type=str, default="Department", help="اسم عمود الهدف (القسم)")
    parser.add_argument("--ignore", type=str, nargs="*", default=["Disease"], help="أعمدة تُستثنى من الميزات")
    parser.add_argument("--repeats", type=int, default=3, help="عدد مرات تبديل الميزة لحساب الأهمية")
    parser.add_argument("--include_derived", action="store_true", help="تضمين الأعمدة المشتقة في قياس الأهمية")
    parser.add_argument("--out", type=str, default="per_class_feature_importance.xlsx", help="اسم ملف الإخراج")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"❌ لم يتم العثور على الحزمة: {args.pkl}")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"❌ لم يتم العثور على ملف البيانات: {args.csv}")
        sys.exit(1)

    with open(args.pkl, "rb") as f:
        PKL = pickle.load(f)

    le = PKL.get("label_encoder")
    feature_cols_training = PKL.get("feature_cols", [])
    n_classes = len(getattr(le, "classes_", []))
    rf_cal = PKL.get("rf_cal")
    gb_cal = PKL.get("gb_cal")
    et_cal = PKL.get("et_cal")
    stack = PKL.get("stack")
    best_w = PKL.get("best_weights", {})
    extra = PKL.get("extra_estimators", [])

    models_for_blend = {"rf": rf_cal, "gb": gb_cal, "et": et_cal}
    for name, mdl in extra:
        models_for_blend[name] = mdl

    df = pd.read_csv(args.csv, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    X, y_str, base_features = build_features_like_training(df, feature_cols_training, args.target, args.ignore)
    y_idx = le.transform(y_str)

    if args.include_derived:
        features_to_eval = list(X.columns)
    else:
        features_to_eval = [c for c in X.columns if c not in DERIVED_COLS and c in feature_cols_training]

    # دوال الاحتمالات لكل نموذج
    proba_funcs = {
        "RandomForest": lambda X_: proba_in_le_order(rf_cal, X_, n_classes, le) if rf_cal else np.zeros((X_.shape[0], n_classes)),
        "GradientBoosting": lambda X_: proba_in_le_order(gb_cal, X_, n_classes, le) if gb_cal else np.zeros((X_.shape[0], n_classes)),
        "ExtraTrees": lambda X_: proba_in_le_order(et_cal, X_, n_classes, le) if et_cal else np.zeros((X_.shape[0], n_classes)),
        "Stacking": lambda X_: proba_in_le_order(stack, X_, n_classes, le) if stack else np.zeros((X_.shape[0], n_classes)),
        "WeightedEnsemble": lambda X_: blend_proba(X_, models_for_blend, best_w, n_classes, le),
    }

    results = {}
    for name, fn in proba_funcs.items():
        print(f"\n⚡ حساب الأهمية: {name}")
        results[name] = compute_permutation_importance_per_class(
            X, y_idx, n_classes, fn, n_repeats=args.repeats, features_to_eval=features_to_eval
        )

    # حفظ إلى Excel
    label_names = list(le.classes_)
    writer = pd.ExcelWriter(args.out, engine="openpyxl")
    for model_name, df_imp in results.items():
        df_out = df_imp.copy()
        df_out["class_name"] = df_out["class_idx"].apply(lambda i: label_names[i] if 0 <= i < len(label_names) else str(i))
        df_out = df_out.sort_values(["class_idx", "importance"], ascending=[True, False])
        df_out.to_excel(writer, index=False, sheet_name=model_name[:31])
    writer.close()
    print(f"\n✅ تم إنشاء الملف: {args.out}")

if __name__ == "__main__":
    main()
