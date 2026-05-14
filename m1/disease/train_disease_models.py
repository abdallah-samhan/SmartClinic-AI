
# -*- coding: utf-8 -*-
"""
train_disease_models.py
يدرب مصنفات الأمراض لكل قسم بشكل هرمي (Department -> Disease) مع مزج وزني للأحتمالات
ويحفظ كل نماذج الأقسام في حزمة واحدة (disease_models.pkl).

المدخلات المتوقعة:
- CSV يحوي على الأقل عمودين: Department (القسم) و Disease (المرض)، وباقي الأعمدة هي ميزات/أعراض رقمية (0/1).
- نفس منطق اشتقاق الميزات المستخدم في تدريب القسم (count_positive, pct_positive, مؤشرات المجموعات).

المخرجات:
- disease_models.pkl : قاموس يحوي:
    {
      'feature_cols': [...],
      'group_defs': {...},
      'derived_cols': [...],
      'departments': {
          dept_name: {
              'label_encoder': LabelEncoder للأمراض,
              'feature_cols': [...],
              'rf_cal', 'gb_cal', 'et_cal', 'stack' (قد تكون None),
              'best_weights': {'rf':..,'gb':..,'et':.., وربما أسماء إضافية},
              'extra_estimators': [(name, estimator), ...],
              'classes_': [أسماء الأمراض بالترتيب]
          },
          ...
      }
    }

الاستخدام (أمثلة):
python train_disease_models.py --csv data.csv --dept_col Department --disease_col Disease --out disease_models.pkl
python train_disease_models.py --csv data.csv --dept_col Department --disease_col Disease --min_per_class 8 --n_splits 5 --seed 42
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# الرسم غير مطلوب هنا، فقط تدريب وحفظ

# ===== تعريفات مجموعات الأعراض (كما في سكربت القسم) =====
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

def add_group_index(X: pd.DataFrame, features, name: str, keywords):
    cols = np.array(features)
    mask = np.zeros(len(cols), dtype=bool)
    for kw in keywords:
        mask |= np.array([kw.lower() in c.lower() for c in cols])
    X[name] = X.loc[:, cols[mask]].sum(axis=1) if mask.any() else 0.0
    return X

def build_features_like_training(df: pd.DataFrame, feature_cols_training=None,
                                 target_cols=None, ignore_cols=None):
    """
    يعيد بناء الميزات (الأصلية + المشتقة) بنفس منطق التدريب المستخدم في القسم.
    إذا كانت feature_cols_training = None سيتم توليد الأعمدة من البيانات ثم تثبيتها.
    """
    target_cols = target_cols or []
    ignore_cols = (ignore_cols or []) + target_cols
    base_features = [c for c in df.columns if c not in ignore_cols]
    X = df[base_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # مشتقات عامة
    X["count_positive"] = X.sum(axis=1)
    X["pct_positive"] = X["count_positive"] / (len(base_features) if len(base_features) else 1)

    # مؤشرات المجموعات
    for gname, kws in GROUPS.items():
        X = add_group_index(X, base_features, gname, kws)

    # في أول مرة لا نملك feature_cols_training: نثبّت الأعمدة الآن
    if feature_cols_training is None:
        feature_cols_training = list(X.columns)

    # ضمان وجود كل الأعمدة وبالترتيب
    for c in feature_cols_training:
        if c not in X.columns:
            X[c] = 0.0
    X = X.loc[:, feature_cols_training]
    return X, feature_cols_training

def proba_in_label_encoder_order(model, X: pd.DataFrame, n_classes: int, le: LabelEncoder):
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

def blend_proba(probas: dict, weights: dict, n_classes: int):
    """probas: dict name-> np.array(N, C), weights: name->float"""
    total_w = 0.0
    out = None
    for name, P in probas.items():
        w = float(weights.get(name, 0.0))
        if P is None or w <= 0:
            continue
        out = P if out is None else (out + w * P)
        total_w += w
    if out is None:
        return np.zeros((next(iter(probas.values())).shape[0], n_classes), dtype=float)
    return out / max(total_w, 1e-9)

def weight_grid(step=0.25, names=("rf","gb","et")):
    vals = np.arange(0, 1.0 + 1e-9, step)
    for ws in product(vals, repeat=len(names)):
        if sum(ws) == 0:
            continue
        yield {n: w for n, w in zip(names, ws)}

def make_base_estimators(seed: int):
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=seed, n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=seed, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        random_state=seed  # لا يدعم class_weight مباشرة
    )
    return rf, gb, et

def choose_calibration_method(n_samples: int):
    # isotonic يحتاج عينات أكثر؛ sigmoid أكثر استقرارًا مع القِلة
    return "isotonic" if n_samples >= 100 else "sigmoid"

def cv_search_weights_for_dept(X, y, seed=42, n_splits=5, step=0.25, verbose=False):
    """
    يبحث عن أفضل أوزان (rf/gb/et) داخل قسم واحد وفق F1_macro على تحقق متقاطع.
    - تدريب النماذج على train-fold فقط
    - التنبؤ على val-fold (بدون معايرة أثناء CV لتبسيط البحث)
    - اختيار الأوزان التي تعظم متوسط F1_macro عبر الطيات
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model_names = ("rf","gb","et")
    weight_scores = defaultdict(list)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        rf, gb, et = make_base_estimators(seed + fold)
        rf.fit(Xtr, ytr)
        gb.fit(Xtr, ytr)
        et.fit(Xtr, ytr)

        # احتمالات غير مُعايرة لأغراض CV
        n_classes = len(np.unique(y))
        le_tmp = LabelEncoder().fit(y)  # لضمان محاذاة الاحتمالات
        probas = {
            "rf": proba_in_label_encoder_order(rf, Xva, n_classes, le_tmp),
            "gb": proba_in_label_encoder_order(gb, Xva, n_classes, le_tmp),
            "et": proba_in_label_encoder_order(et, Xva, n_classes, le_tmp)
        }

        for W in weight_grid(step=step, names=model_names):
            P = blend_proba(probas, W, n_classes)
            yhat = np.argmax(P, axis=1)
            f1m = f1_score(yva, yhat, average="macro")
            weight_scores[tuple(W[n] for n in model_names)].append(f1m)

    # متوسط الأداء عبر الطيات
    best_w = None
    best_mean = -1
    for ws, scores in weight_scores.items():
        mean_f1 = float(np.mean(scores)) if scores else 0.0
        if mean_f1 > best_mean:
            best_mean = mean_f1
            best_w = {name: w for name, w in zip(model_names, ws)}

    if verbose:
        print(f"[CV] أفضل أوزان: {best_w} | F1_macro_cv={best_mean:.4f}")

    return best_w or {"rf":1.0,"gb":0.0,"et":0.0}, best_mean

def train_calibrated_models_full(X, y, seed=42, calib="sigmoid"):
    """تدريب النماذج على كامل بيانات القسم + معايرة الاحتمالات."""
    rf, gb, et = make_base_estimators(seed)
    rf.fit(X, y); et.fit(X, y); gb.fit(X, y)

    # Calibrate
    rf_cal = CalibratedClassifierCV(rf, method=calib, cv=3)
    gb_cal = CalibratedClassifierCV(gb, method=calib, cv=3)
    et_cal = CalibratedClassifierCV(et, method=calib, cv=3)
    rf_cal.fit(X, y); gb_cal.fit(X, y); et_cal.fit(X, y)

    return rf_cal, gb_cal, et_cal

def main():
    ap = argparse.ArgumentParser(description="تدريب نماذج الأمراض داخل كل قسم مع مزج وزني")
    ap.add_argument("--csv", type=str, required=True, help="ملف البيانات")
    ap.add_argument("--dept_col", type=str, default="Department", help="اسم عمود القسم")
    ap.add_argument("--disease_col", type=str, default="Disease", help="اسم عمود المرض")
    ap.add_argument("--ignore", nargs="*", default=[], help="أعمدة يتم تجاهلها من الميزات (بخلاف القسم/المرض)")
    ap.add_argument("--out", type=str, default="disease_models.pkl", help="اسم ملف الحزمة الناتجة")
    ap.add_argument("--n_splits", type=int, default=5, help="عدد طيات التحقق المتقاطع داخل القسم")
    ap.add_argument("--min_per_class", type=int, default=5, help="أدنى عدد عينات لكل مرض للإبقاء عليه")
    ap.add_argument("--seed", type=int, default=42, help="Seed")
    ap.add_argument("--weight_step", type=float, default=0.25, help="خطوة شبكة الأوزان (0.25 توفّر سرعة جيدة)")
    ap.add_argument("--verbose", action="store_true", help="طباعة تفاصيل إضافية")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"لا يوجد الملف: {args.csv}")

    # قراءة البيانات
    df = pd.read_csv(args.csv, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    for col in (args.dept_col, args.disease_col):
        if col not in df.columns:
            raise ValueError(f"عمود '{col}' غير موجود في CSV.")

    # إعداد الميزات (تثبيت feature_cols مرة واحدة ليستخدمها كل قسم)
    X_all, feature_cols_fixed = build_features_like_training(
        df, feature_cols_training=None,
        target_cols=[args.dept_col, args.disease_col],
        ignore_cols=args.ignore
    )

    # الحزمة النهائية
    PKL = {
        "feature_cols": feature_cols_fixed,
        "group_defs": GROUPS,
        "derived_cols": DERIVED_COLS,
        "departments": {}
    }

    # تدريب لكل قسم
    departments = sorted(df[args.dept_col].astype(str).unique())
    print(f"🧭 عدد الأقسام: {len(departments)} -> {departments}")

    for dept in departments:
        dfd = df[df[args.dept_col].astype(str) == str(dept)].copy()
        y_disease_str = dfd[args.disease_col].astype(str)

        # تصفية الأمراض قليلة العينات (تحسين الاستقرار)
        counts = y_disease_str.value_counts()
        keep = counts[counts >= args.min_per_class].index
        dfd = dfd[y_disease_str.isin(keep)].copy()
        if dfd.shape[0] == 0:
            print(f"⚠️ القسم '{dept}': لا توجد عينات كافية بعد التصفية. سيتم تخطيه.")
            continue

        y_disease_str = dfd[args.disease_col].astype(str)
        # استخدم نفس feature_cols_fixed لضمان الاتساق
        Xd = dfd.drop(columns=[args.dept_col, args.disease_col] + args.ignore, errors="ignore")
        Xd = Xd.reindex(columns=[c for c in X_all.columns if c in Xd.columns], fill_value=0)
        # أعد البناء بنفس المنطق لكي لا نفقد المشتقات
        Xd, _ = build_features_like_training(
            dfd, feature_cols_training=feature_cols_fixed,
            target_cols=[args.dept_col, args.disease_col],
            ignore_cols=args.ignore
        )

        le = LabelEncoder()
        y = le.fit_transform(y_disease_str.values)
        classes = list(le.classes_)
        n_classes = len(classes)
        n_samples = len(y)

        print(f"\n===== قسم: {dept} | عينات: {n_samples} | أمراض: {n_classes} =====")
        if n_classes < 2:
            # حالة فئة واحدة: نخزن مخمنًا ثابتًا (will always predict that disease)
            only = classes[0]
            print(f"⚠️ القسم '{dept}': مرض واحد فقط بعد التصفية ({only}). سيتم تخزين مُتنبئ ثابت.")
            PKL["departments"][dept] = {
                "label_encoder": le,
                "feature_cols": feature_cols_fixed,
                "rf_cal": None, "gb_cal": None, "et_cal": None, "stack": None,
                "best_weights": {"rf":0.0,"gb":0.0,"et":0.0},
                "extra_estimators": [],
                "classes_": classes,
                "single_class": True,
                "single_class_label": only
            }
            continue

        # اختيار طريقة المعايرة بحسب حجم القسم
        calib = choose_calibration_method(n_samples)

        # بحث الأوزان عبر تحقق متقاطع
        splits = min(args.n_splits, np.min(np.bincount(y)))  # لا تتجاوز أقل حجم فئة
        splits = max(3, min(splits, args.n_splits))          # من 3 إلى n_splits
        best_w, cv_mean = cv_search_weights_for_dept(
            Xd, y, seed=args.seed, n_splits=splits, step=args.weight_step, verbose=args.verbose
        )

        # تدريب كامل + معايرة
        rf_cal, gb_cal, et_cal = train_calibrated_models_full(Xd, y, seed=args.seed, calib=calib)

        # خزن القسم
        PKL["departments"][dept] = {
            "label_encoder": le,
            "feature_cols": feature_cols_fixed,
            "rf_cal": rf_cal, "gb_cal": gb_cal, "et_cal": et_cal, "stack": None,
            "best_weights": best_w,
            "extra_estimators": [],   # يمكن إضافة نماذج إضافية إن رغبت مستقبلاً
            "classes_": classes,
            "single_class": False,
            "cv_f1_macro": float(cv_mean),
            "calibration": calib
        }

        # طباعة موجز سريع للأداء على كامل بيانات القسم (تقدير متفائل)
        from sklearn.metrics import f1_score
        def _proba_in_order(mdl): return proba_in_label_encoder_order(mdl, Xd, n_classes, le)
        probas = {"rf": _proba_in_order(rf_cal), "gb": _proba_in_order(gb_cal), "et": _proba_in_order(et_cal)}
        P = blend_proba(probas, best_w, n_classes)
        yhat = np.argmax(P, axis=1)
        f1m_full = f1_score(y, yhat, average="macro")
        print(f"✅ {dept}: أفضل أوزان {best_w} | F1_macro_CV={cv_mean:.3f} | F1_macro_full={f1m_full:.3f} | calib={calib}")

    # حفظ الحزمة
    with open(args.out, "wb") as f:
        pickle.dump(PKL, f)
    print(f"\n💾 تم حفظ الحزمة: {args.out}")
    print("🎉 تم الانتهاء من تدريب نماذج الأمراض لكل قسم.")

if __name__ == "__main__":
    main()
