# -*- coding: utf-8 -*-
"""
report_weighted_ensemble.py
تقرير شامل لأداء المزج الوزني والنماذج الفردية بناءً على ملفات الناتجة من m1.py

المخرجات:
- performance_report.xlsx : جداول نتائج إجمالية وتفصيلية لكل نموذج
- classification_report_weighted.csv : تقرير تصنيف لكل فئة للمزج الوزني
- confusion_matrix_best.png : مصفوفة الالتباس لأفضل نموذج حسب F1_macro
- (اختياري) تبويب Meta داخل الإكسل: دعم الفئات وعدد التنبؤات الأساسية وF1 لكل فئة
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)

# رسم بدون واجهة تفاعلية
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== تعريف مجموعات الأعراض (مثل التدريب) =====
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
def add_group_index(X: pd.DataFrame, features, name: str, keywords):
    cols = np.array(features)
    mask = np.zeros(len(cols), dtype=bool)
    for kw in keywords:
        mask |= np.array([kw.lower() in c.lower() for c in cols])
    if mask.any():
        X[name] = X.loc[:, cols[mask]].sum(axis=1)
    else:
        X[name] = 0.0
    return X

def build_features_like_training(df: pd.DataFrame, feature_cols_training, target_col: str, ignore_cols: list):
    base_features = [c for c in df.columns if c not in ignore_cols + [target_col]]
    X = df[base_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X['count_positive'] = X.sum(axis=1)
    X['pct_positive'] = X['count_positive'] / (len(base_features) if len(base_features) else 1)
    for gname, kws in GROUPS.items():
        X = add_group_index(X, base_features, gname, kws)
    for c in feature_cols_training:
        if c not in X.columns:
            X[c] = 0.0
    X = X.loc[:, feature_cols_training]
    return X

def proba_in_le_order(model, X, n_classes, le):
    if model is None:
        return np.zeros((len(X), n_classes))
    proba = np.asarray(model.predict_proba(X))
    aligned = np.zeros((len(X), n_classes))
    est_classes = getattr(model, "classes_", None)
    if est_classes is None:
        aligned[:, :proba.shape[1]] = proba
        return aligned
    est_classes = np.asarray(est_classes)
    if np.issubdtype(est_classes.dtype, np.integer):
        aligned[:, est_classes] = proba
    else:
        idx = le.transform(est_classes.astype(str))
        aligned[:, idx] = proba
    return aligned

def blend_proba(X, models_for_blend: dict, weights: dict, n_classes: int, le):
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

# ===== البرنامج الرئيسي =====
def main():
    parser = argparse.ArgumentParser(description="تقرير أداء المزج الوزني والنماذج الفردية")
    parser.add_argument('--pkl', default='m1/dept_models.pkl', type=str)
    parser.add_argument('--csv', required=True, type=str)
    parser.add_argument('--target', default='Department', type=str)
    parser.add_argument('--ignore', nargs='*', default=['Disease'])
    parser.add_argument('--out', default='performance_report.xlsx', type=str)
    parser.add_argument('--include_meta', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        raise FileNotFoundError(f"لم يتم العثور على الحزمة: {args.pkl}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"لم يتم العثور على ملف البيانات: {args.csv}")

    with open(args.pkl, 'rb') as f:
        PKL = pickle.load(f)

    le = PKL.get('label_encoder')
    feature_cols = PKL.get('feature_cols')
    if le is None or feature_cols is None:
        raise ValueError("الحزمة لا تحتوي على label_encoder أو feature_cols")

    labels = list(le.classes_)
    n_classes = len(labels)
    rf = PKL.get('rf_cal')
    gb = PKL.get('gb_cal')
    et = PKL.get('et_cal')
    stack = PKL.get('stack')
    best_w = PKL.get('best_weights', {})
    extra = PKL.get('extra_estimators', [])

    models_for_blend = {'rf': rf, 'gb': gb, 'et': et}
    for name, mdl in extra:
        models_for_blend[name] = mdl

    df = pd.read_csv(args.csv, encoding='utf-8', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    if args.target not in df.columns:
        raise ValueError(f"عمود الهدف '{args.target}' غير موجود في CSV.")
    y_true_str = df[args.target].astype(str)
    X = build_features_like_training(df, feature_cols, args.target, args.ignore)
    y_true = le.transform(y_true_str)

    proba_funcs = {}
    if rf is not None: proba_funcs['RandomForest'] = lambda: proba_in_le_order(rf, X, n_classes, le)
    if gb is not None: proba_funcs['GradientBoosting'] = lambda: proba_in_le_order(gb, X, n_classes, le)
    if et is not None: proba_funcs['ExtraTrees'] = lambda: proba_in_le_order(et, X, n_classes, le)
    if stack is not None: proba_funcs['Stacking'] = lambda: proba_in_le_order(stack, X, n_classes, le)
    proba_funcs['WeightedEnsemble'] = lambda: blend_proba(X, models_for_blend, best_w, n_classes, le)

    rows = []
    per_model_reports = {}
    for name, fn in proba_funcs.items():
        proba = fn()
        y_pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
        per_model_reports[name] = pd.DataFrame(report).transpose().reset_index().rename(columns={'index':'label'})

        rows.append({
            'model': name,
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro
        })

    df_summary = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)

    # تبويب Meta
    meta_df = None
    if args.include_meta:
        y_pred_w = np.argmax(proba_funcs['WeightedEnsemble'](), axis=1)
        f1_base_per_class = []
        for k in range(n_classes):
            y_true_k = (y_true==k).astype(int)
            y_pred_k = (y_pred_w==k).astype(int)
            try:
                f1_base_per_class.append(f1_score(y_true_k, y_pred_k))
            except:
                f1_base_per_class.append(0.0)
        meta_rows = []
        for k in range(n_classes):
            meta_rows.append({
                "class_idx": k,
                "class_name": labels[k],
                "support_true": int((y_true==k).sum()),
                "pred_count_weighted_base": int((y_pred_w==k).sum()),
                "f1_base_weighted": float(f1_base_per_class[k])
            })
        meta_df = pd.DataFrame(meta_rows)

    # حفظ Excel
    with pd.ExcelWriter(args.out, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False, sheet_name='Summary')
        for name, df_rep in per_model_reports.items():
            sheet = name[:31]
            df_rep.to_excel(writer, index=False, sheet_name=sheet)
        if meta_df is not None:
            meta_df.to_excel(writer, index=False, sheet_name='Meta')

    print(f"✅ تم إنشاء التقرير: {args.out}")

    # مصفوفة الالتباس لأفضل نموذج
    best_name = df_summary.iloc[0]['model']
    proba_best = proba_funcs[best_name]()
    y_pred_best = np.argmax(proba_best, axis=1)
    cm = confusion_matrix(y_true, y_pred_best, labels=np.arange(n_classes))

    plt.figure(figsize=(12,10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {best_name}', fontsize=14)
    plt.colorbar()
    ticks = np.arange(n_classes)
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig('confusion_matrix_best.png', dpi=160)
    plt.close()
    print('🖼️ Saved confusion_matrix_best.png')

    # CSV تقرير المزج الوزني
    y_pred_w = np.argmax(proba_funcs['WeightedEnsemble'](), axis=1)
    rep_w = classification_report(y_true, y_pred_w, target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame(rep_w).transpose().to_csv('classification_report_weighted.csv', encoding='utf-8', index=True)
    print('📄 Saved classification_report_weighted.csv')


if __name__ == '__main__':
    main()
