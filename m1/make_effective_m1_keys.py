
# make_effective_m1_keys.py
# يولّد m1/effective_m1_keys.txt من feature_importance_rf_et.csv مع التقاطع مع feature_cols في dept_models.pkl
# يدعم حالات: أسماء الميزات في فهرس CSV أو في عمود (feature / Unnamed: 0)، ويطبّع الأسماء لضمان التقاطع.

import os, pickle
import pandas as pd

BASE = r"C:\xampp\htdocs\SmartClinic\proj"
PKL_PATH = os.path.join(BASE, "m1", "dept_models.pkl")
IMP_CANDIDATES = [
    os.path.join(BASE, "m1", "feature_importance_rf_et.csv"),
    os.path.join(BASE, "feature_importance_rf_et.csv"),
]
OUT_PATH = os.path.join(BASE, "m1", "effective_m1_keys.txt")
TOP_N = 50  # عدّل الرقم حسب رغبتك

DERIVED = {
    "count_positive","pct_positive",
    "respiratory_idx","cardiac_idx","neurology_idx","ent_idx",
    "derm_idx","gi_idx","uro_idx","heme_idx","endocrine_idx","immune_idx"
}

def norm(s: str) -> str:
    """تطبيع اسم الميزة للمطابقة: حروف صغيرة + إزالة المسافات/الشرطات/الـ underscores."""
    return (s or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")

# 1) حمل قائمة ميزات التدريب من الحزمة (مع استبعاد المشتقات)
print("[DEBUG] Loading PKL:", PKL_PATH, "| Exists:", os.path.exists(PKL_PATH))
with open(PKL_PATH, "rb") as f:
    PKL = pickle.load(f)

feature_cols = PKL.get("feature_cols", [])
base_cols = [c for c in feature_cols if c not in DERIVED]
if not base_cols:
    raise RuntimeError("feature_cols فارغة أو لا تحتوي أعمدة أساسية بعد استبعاد المشتقات.")

# خرائط أسماء أصلية ← مطبّعة
base_norm_map = {norm(c): c for c in base_cols}
base_norm_set = set(base_norm_map.keys())
print(f"[DEBUG] Base training features: {len(base_cols)}")

# 2) حمّل CSV للأهميات
imp_path = next((p for p in IMP_CANDIDATES if os.path.exists(p)), None)
if not imp_path:
    raise RuntimeError("تعذّر العثور على feature_importance_rf_et.csv في المواقع المتوقعة.")
print("[DEBUG] Loading importance CSV:", imp_path)

imp = pd.read_csv(imp_path)
imp.columns = [c.strip() for c in imp.columns]

# ابحث عن أعمدة RF/ET (حسّاس/غير حسّاس للحروف)
def find_col(df, name):
    for c in df.columns:
        if c.strip().lower() == name:
            return c
    return None

rf_col = find_col(imp, "rf_importance")
et_col = find_col(imp, "et_importance")
if rf_col is None or et_col is None:
    raise RuntimeError("لم يتم العثور على أعمدة RF_importance/ET_importance في CSV.")

# حدّد عمود أسماء الميزات إن لم يكن في الفهرس
feat_col = None
# مرشّح شائع: 'feature' أو 'Unnamed: 0' أو أول عمود غير RF/ET
for cand in ["feature", "Feature", "FEATURE", "Unnamed: 0", "unnamed: 0"]:
    if cand in imp.columns:
        feat_col = cand
        break
if feat_col is None:
    # خذ أول عمود ليس RF/ET (إن وجد)
    non_metric_cols = [c for c in imp.columns if c not in [rf_col, et_col]]
    if len(non_metric_cols) >= 1:
        feat_col = non_metric_cols[0]

# استخرج أسماء الميزات من الفهرس أو العمود
if feat_col and feat_col in imp.columns:
    feature_names = imp[feat_col].astype(str).tolist()
    print(f"[DEBUG] Using feature name column: {feat_col}")
else:
    # افترض أن أسماء الميزات في الفهرس (index) — أحيانًا تُحفظ كسطر 'Unnamed: 0' عند القراءة
    if imp.index.dtype == "object":
        feature_names = imp.index.astype(str).tolist()
        print("[DEBUG] Using CSV index as feature names.")
    else:
        # محاولة أخيرة: إن وجد عمود 'Unnamed: 0' بعد القراءة
        if "Unnamed: 0" in imp.columns:
            feature_names = imp["Unnamed: 0"].astype(str).tolist()
            print("[DEBUG] Using 'Unnamed: 0' column as feature names.")
        else:
            raise RuntimeError("تعذّر تحديد عمود/فهرس أسماء الميزات في CSV.")

# 3) احسب مجموع الأهمية ورتّب
imp["sum_imp"] = imp[rf_col].fillna(0.0) + imp[et_col].fillna(0.0)

# اربط كل صف باسم ميزة مطبّع
imp["_feat_norm"] = pd.Series([norm(x) for x in feature_names], index=imp.index)

# 4) انتقِ Top-N ثم تقاطع مطبّع مع ميزات التدريب
imp_sorted = imp.sort_values("sum_imp", ascending=False)
ordered_norm = imp_sorted["_feat_norm"].tolist()

# التقاطع (مطبّع)، ثم أعد الأسماء الأصلية من training
effective_norm = []
seen = set()
for n in ordered_norm:
    if n in base_norm_set and n not in seen:
        effective_norm.append(n)
        seen.add(n)
    if len(effective_norm) >= TOP_N:
        break

if not effective_norm:
    # Fallback آمن: خذ أول TOP_N من ميزات التدريب الأساسية مباشرةً
    print("[WARN] لا يوجد تقاطع بين CSV وميزات التدريب بعد التطبيع. سيتم استخدام أول TOP_N من قائمة التدريب الأساسية.")
    effective = base_cols[:TOP_N]
else:
    effective = [base_norm_map[n] for n in effective_norm]

# 5) احفظ النتيجة في m1/effective_m1_keys.txt
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for k in effective:
        f.write(k + "\n")

print(f"[DONE] تم إنشاء {OUT_PATH} بعدد ميزات: {len(effective)}")
