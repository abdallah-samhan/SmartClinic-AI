
# -*- coding: utf-8 -*-
"""
High-Accuracy Pipeline (Fixed Labels):
- تشفير الفئات بأعداد صحيحة (LabelEncoder) واستخدامها في كل التدريب والتقييم
- SMOTE(k=1) مع تحويل تلقائي إلى RandomOverSampler عند الحاجة
- ضبط معاملات واسع مع 5-طيّات CV
- معايرة الاحتمالات (Calibration)
- Stacking + Ensemble بوزن ديناميكي
- Checkpointing & Logging
"""

import os, time, pickle, warnings, json, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
import matplotlib.pyplot as plt

# ---------------------- إعدادات عامة ----------------------
CSV_PATH         = "grouped_by_department_merged.csv"
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
N_JOBS           = -1
USE_SMOTE        = True
CV_FOLDS         = 5
CALIBRATE_PROBS  = True
CALIB_METHOD     = "sigmoid"

RF_N_ITER = 20
GB_N_ITER = 20
ET_N_ITER = 16

WEIGHT_GRID = [0.25, 0.5, 0.75, 1.0, 1.25]
WEIGHT_RANDOM_TRIALS = 200

CKPT_RF  = "rf_best.pkl"
CKPT_GB  = "gb_best.pkl"
CKPT_ET  = "et_best.pkl"
CKPT_LOG = "training_log.json"

# SMOTE/ROS
SMOTE_AVAILABLE = False
ROS_AVAILABLE   = False
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        SMOTE_AVAILABLE = True
        ROS_AVAILABLE   = True
    except Exception:
        pass

# نماذج تعزيز اختيارية
XGB_AVAILABLE = False
LGB_AVAILABLE = False
CAT_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    pass
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    pass

# ---------------------- أدوات مساعدة ----------------------
def log_event(tag, info):
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "info": info}
    try:
        if os.path.exists(CKPT_LOG):
            data = json.loads(open(CKPT_LOG, "r", encoding="utf-8").read())
        else:
            data = []
        data.append(entry)
        open(CKPT_LOG, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass
    print(f"[{entry['time']}] {tag}: {info}")

def save_ckpt(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# محاذاة احتمالات النموذج إلى ترتيب الفئات المشفّرة [0..n_classes-1]
def proba_in_le_order(model, X, n_classes, le):
    proba = model.predict_proba(X)
    est_classes = getattr(model, "classes_", None)
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    if est_classes is None:
        # نفترض أن الأعمدة بالفعل بترتيب 0..n_classes-1
        aligned[:, :proba.shape[1]] = proba
        return aligned
    est_classes = np.array(est_classes)
    # تحويل الفئات (سواء كانت نصية أو عددية) إلى فهارس LabelEncoder
    if est_classes.dtype.kind in {"U", "S", "O"}:  # نصية
        est_idx = le.transform(est_classes)
    else:
        est_idx = est_classes.astype(int)
    aligned[:, est_idx] = proba
    return aligned

# ---------------------- قراءة البيانات ----------------------
t0 = time.time()
df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
df.columns = [c.strip() for c in df.columns]

TARGET  = "Department"
IGNORE  = ["Disease"]
FEATURES = [c for c in df.columns if c not in IGNORE + [TARGET]]

X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[TARGET].astype(str)
log_event("DATA", f"Loaded {df.shape[0]} rows, {len(FEATURES)} features. Classes={y.nunique()}")

# ---------------------- هندسة الميزات ----------------------
X["count_positive"] = X.sum(axis=1)
X["pct_positive"]   = X["count_positive"] / (len(FEATURES) if len(FEATURES) else 1)

def add_group_index(X, features, name, keywords):
    cols = np.array(features)
    mask = np.zeros(len(cols), dtype=bool)
    for kw in keywords:
        mask |= np.array([kw.lower() in c.lower() for c in cols])
    X[name] = X.loc[:, cols[mask]].sum(axis=1)
    return X

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
for gname, kws in GROUPS.items():
    X = add_group_index(X, FEATURES, gname, kws)

# ---------------------- معالجة عدم التوازن ----------------------
if SMOTE_AVAILABLE:
    try:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=1)
        X_res, y_res = sm.fit_resample(X, y)
        log_event("SMOTE", f"Applied SMOTE(k=1). New size={X_res.shape[0]}")
    except ValueError as e:
        log_event("SMOTE_FAIL", str(e))
        if ROS_AVAILABLE:
            ros = RandomOverSampler(random_state=RANDOM_STATE)
            X_res, y_res = ros.fit_resample(X, y)
            log_event("ROS", f"Applied RandomOverSampler. New size={X_res.shape[0]}")
        else:
            vc = y.value_counts()
            max_count = vc.max()
            X_parts, y_parts = [], []
            for cls, cnt in vc.items():
                idx = np.where(y.values == cls)[0]
                sampled_idx = np.random.choice(idx, size=max_count, replace=True)
                X_parts.append(X.iloc[sampled_idx])
                y_parts.append(pd.Series(y.iloc[sampled_idx].values))
            X_res = pd.concat(X_parts, axis=0).reset_index(drop=True)
            y_res = pd.concat(y_parts, axis=0).reset_index(drop=True)
            log_event("OVERSAMPLE_MANUAL", f"Manual oversampling. New size={X_res.shape[0]}")
else:
    vc = y.value_counts()
    max_count = vc.max()
    X_parts, y_parts = [], []
    for cls, cnt in vc.items():
        idx = np.where(y.values == cls)[0]
        sampled_idx = np.random.choice(idx, size=max_count, replace=True)
        X_parts.append(X.iloc[sampled_idx])
        y_parts.append(pd.Series(y.iloc[sampled_idx].values))
    X_res = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_res = pd.concat(y_parts, axis=0).reset_index(drop=True)
    log_event("OVERSAMPLE_MANUAL", f"Manual oversampling. New size={X_res.shape[0]}")

# ---------------------- تقسيم + LabelEncoder عالمي ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=TEST_SIZE, stratify=y_res, random_state=RANDOM_STATE
)
le = LabelEncoder().fit(y_res)
y_train_enc = le.transform(y_train)   # أعداد صحيحة لكل الفئات
y_test_enc  = le.transform(y_test)
n_classes   = len(le.classes_)
log_event("SPLIT", f"Train={X_train.shape[0]}, Test={X_test.shape[0]} | n_classes={n_classes}")

# ---------------------- النماذج الأساسية (تعمل على y_enc) ----------------------
rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced_subsample")
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
et = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced")

# ---------------------- ضبط معاملات مع Checkpoint ----------------------
def tune_or_load(name, model, param_dist, n_iter):
    ckpt_path = {"rf": CKPT_RF, "gb": CKPT_GB, "et": CKPT_ET}[name]
    if os.path.exists(ckpt_path):
        best = load_ckpt(ckpt_path)
        log_event("CKPT_LOAD", f"Loaded {name} best params from checkpoint.")
        return best
    t_start = time.time()
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, scoring="f1_macro",
                                cv=CV_FOLDS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    search.fit(X_train, y_train_enc)
    best = search.best_estimator_
    save_ckpt(ckpt_path, best)
    log_event("TUNE", f"{name} tuned in {time.time()-t_start:.1f}s -> {search.best_params_}")
    return best

rf_best = tune_or_load(
    "rf", rf,
    {"n_estimators":[400,800,1200],
     "max_depth":[None,20,30,40],
     "min_samples_split":[2,5,10],
     "min_samples_leaf":[1,2,4],
     "max_features":["sqrt","log2",0.5]},
    n_iter=RF_N_ITER
)
gb_best = tune_or_load(
    "gb", gb,
    {"n_estimators":[200,400,800],
     "learning_rate":[0.01,0.05,0.1],
     "max_depth":[2,3,4],
     "subsample":[0.8,1.0]},
    n_iter=GB_N_ITER
)
et_best = tune_or_load(
    "et", et,
    {"n_estimators":[600,900,1200],
     "max_depth":[None,30,50],
     "min_samples_split":[2,4,8],
     "min_samples_leaf":[1,2,4],
     "max_features":["sqrt","log2",0.5]},
    n_iter=ET_N_ITER
)

# ---------------------- نماذج تعزيز اختيارية (تُدرَّب على y_enc) ----------------------
extra_estimators = []
if XGB_AVAILABLE:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_classes,
        random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    xgb_clf.fit(X_train, y_train_enc)
    extra_estimators.append(("xgb", xgb_clf))
    log_event("XGB", "XGBoost added to ensemble.")

if LGB_AVAILABLE:
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=n_classes,
        random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    lgb_clf.fit(X_train, y_train_enc)
    extra_estimators.append(("lgb", lgb_clf))
    log_event("LGB", "LightGBM added to ensemble.")

if CAT_AVAILABLE:
    cat_clf = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        loss_function="MultiClass", random_state=RANDOM_STATE,
        verbose=False
    )
    cat_clf.fit(X_train, y_train_enc)
    extra_estimators.append(("cat", cat_clf))
    log_event("CAT", "CatBoost added to ensemble.")

# ---------------------- معايرة الاحتمالات ----------------------
def calibrate(name, estimator, y_enc):
    if not CALIBRATE_PROBS:
        return estimator
    t_start = time.time()
    cal = CalibratedClassifierCV(estimator, method=CALIB_METHOD, cv=3)
    cal.fit(X_train, y_enc)
    log_event("CALIB", f"{name} calibrated ({CALIB_METHOD}) in {time.time()-t_start:.1f}s")
    return cal

# نعاير الجميع على y المشفّر
rf_cal = calibrate("RF", rf_best, y_train_enc)
gb_cal = calibrate("GB", gb_best, y_train_enc)
et_cal = calibrate("ET", et_best, y_train_enc)

# ---------------------- Stacking (cv=5) ----------------------
# نستخدم الأساسيات + المعزِّزات الاختيارية
base_estimators = [("rf", rf_cal), ("gb", gb_cal), ("et", et_cal)] + extra_estimators
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=400, class_weight="balanced"),
    stack_method="predict_proba",
    passthrough=False,
    cv=CV_FOLDS
)
t_start = time.time()
stack.fit(X_train, y_train_enc)
log_event("STACK", f"Stacking trained in {time.time()-t_start:.1f}s with {len(base_estimators)} base learners.")

# ---------------------- تجميع بوزن ديناميكي ----------------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def eval_weights(wdict, X_val, models_list, n_classes, le):
    proba_sum = 0.0
    total_w = 0.0
    for key, mdl in models_list.items():
        w = wdict.get(key, 0.0)
        if w <= 0: 
            continue
        proba = proba_in_le_order(mdl, X_val, n_classes, le)
        proba_sum += w * proba
        total_w += w
    proba_sum = proba_sum / max(total_w, 1e-9)
    return proba_sum

models_for_blend = {"rf": rf_cal, "gb": gb_cal, "et": et_cal}
for name, mdl in extra_estimators:
    models_for_blend[name] = mdl

# شبكة أوزان + بحث عشوائي
best_w = None
best_score = -np.inf
keys = list(models_for_blend.keys())

grid_combos = []
def gen_grid(keys, grid):
    grid_combos.append({k:1.0 for k in keys})
    for g in grid:
        grid_combos.append({k:g for k in keys})
    if len(keys) >= 3:
        grid_combos.append({keys[0]:1.25, keys[1]:1.0,  keys[2]:0.75})
        grid_combos.append({keys[0]:0.75, keys[1]:1.25, keys[2]:1.0})
        grid_combos.append({keys[0]:1.0,  keys[1]:0.75, keys[2]:1.25})

gen_grid(keys, WEIGHT_GRID)
for _ in range(WEIGHT_RANDOM_TRIALS):
    wdict = {k: random.choice(WEIGHT_GRID) for k in keys}
    grid_combos.append(wdict)

log_event("BLEND", f"Weight combos to try: {len(grid_combos)}")

t_start = time.time()
for wdict in grid_combos:
    scores = []
    for tr_idx, val_idx in cv.split(X_train, y_train_enc):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_val_enc    = y_train_enc[val_idx]
        proba = eval_weights(wdict, X_val, models_for_blend, n_classes, le)
        pred_idx = np.argmax(proba, axis=1)
        scores.append(f1_score(y_val_enc, pred_idx, average="macro"))
    mean_score = float(np.mean(scores))
    if mean_score > best_score:
        best_score = mean_score
        best_w = wdict.copy()

log_event("BLEND_DONE", f"Best weights {best_w} with CV F1_macro={best_score:.4f} in {time.time()-t_start:.1f}s")

# ---------------------- تقييم على الاختبار ----------------------
def evaluate_model(name, mdl):
    y_pred_idx = mdl.predict(X_test)              # تنبؤات كفئات مشفّرة (أعداد صحيحة)
    return {
        "accuracy": float(accuracy_score(y_test_enc, y_pred_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, y_pred_idx)),
        "f1_macro": float(f1_score(y_test_enc, y_pred_idx, average="macro"))
    }

results = {}
results["RandomForest"]     = evaluate_model("RF", rf_cal)
results["GradientBoosting"] = evaluate_model("GB", gb_cal)
results["ExtraTrees"]       = evaluate_model("ET", et_cal)
results["Stacking"]         = evaluate_model("Stacking", stack)

# Ensemble بالأوزان المثلى
proba_test = eval_weights(best_w, X_test, models_for_blend, n_classes, le)
pred_idx   = np.argmax(proba_test, axis=1)
results["WeightedEnsemble"] = {
    "weights": best_w,
    "accuracy": float(accuracy_score(y_test_enc, pred_idx)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, pred_idx)),
    "f1_macro": float(f1_score(y_test_enc, pred_idx, average="macro"))
}

best_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
cm_pred_idx = (pred_idx if best_name=="WeightedEnsemble"
               else (stack.predict(X_test) if best_name=="Stacking"
                     else (rf_cal.predict(X_test) if best_name=="RandomForest"
                           else gb_cal.predict(X_test) if best_name=="GradientBoosting"
                           else et_cal.predict(X_test))))

# ---------------------- مصفوفة الالتباس (أسماء الفئات) ----------------------
try:
    matplotlib.rcParams["font.size"] = 9
    plt.figure(figsize=(12, 10))
    labels = le.classes_
    cm = confusion_matrix(y_test_enc, cm_pred_idx, labels=np.arange(n_classes))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("مصفوفة الالتباس للنموذج الأفضل")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    plt.close()
    log_event("CM", "Saved confusion_matrix.png")
except Exception as e:
    log_event("CM_FAIL", str(e))

# ---------------------- أهمية الميزات ----------------------
fi_rf = pd.Series(rf_best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
fi_et = pd.Series(et_best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
pd.DataFrame({"RF_importance":fi_rf, "ET_importance":fi_et}).fillna(0).to_csv("feature_importance_rf_et.csv")
log_event("FI", "Saved feature_importance_rf_et.csv")

# ---------------------- حفظ الحزمة ----------------------
with open("dept_models.pkl","wb") as f:
    pickle.dump({
        "rf": rf_best, "gb": gb_best, "et": et_best, "stack": stack,
        "rf_cal": rf_cal, "gb_cal": gb_cal, "et_cal": et_cal,
        "extra_estimators": extra_estimators,
        "label_encoder": le,
        "best_weights": best_w,
        "feature_cols": X_train.columns.tolist()
    }, f)
log_event("SAVE", "Saved dept_models.pkl")

# ---------------------- ملخص نهائي ----------------------
print("\n===== ملخص الأداء (High-Accuracy — Fixed Labels) =====")
for k, v in results.items():
    print(f"{k} -> {v}")
print(f"\nأفضل طريقة: {best_name}")
print("\nتم حفظ الملفات:")
print("- confusion_matrix.png")
print("- feature_importance_rf_et.csv")
print("- dept_models.pkl")
print("- rf_best.pkl / gb_best.pkl / et_best.pkl (إذا توليف لأول مرة)")
print(f"\nالزمن الكلي: {time.time()-t0:.1f} ثانية")
