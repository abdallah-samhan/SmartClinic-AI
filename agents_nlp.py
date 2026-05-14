
# -*- coding: utf-8 -*-
"""
agents_nlp.py (Arabic/English extraction + English negation + external AR synonyms + auto AR synonyms)
نسخة متوافقة مع coverage_runner.py v5 — بعد إضافة ar_synonyms_medical.csv

- يعرّف: NLPAgent, load_symptom_keys, en_phrases_for_key, AR_SYNONYMS
- يحمل مرادفات عربية خارجية افتراضيًا من: ar_synonyms_medical.csv
- top_n=0 افتراضيًا (يشمل جميع الأعراض)
"""

import os, re, csv, unicodedata
from typing import Dict, List, Tuple, Optional

# ==============================
# Arabic normalization helpers
# ==============================
AR_DIacritics = re.compile(r'[\u064B-\u0652]')
AR_TATWEEL = "\u0640"

def normalize_text(text: str) -> str:
    """تطبيع عربي + Lower + إزالة التشكيل والتطويل + توحيد الحروف."""
    if not text: return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = AR_DIacritics.sub("", t).replace(AR_TATWEEL, "")
    t = (t
         .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
         .replace("ة", "ه").replace("ى", "ي"))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ==============================
# English normalization helper
# ==============================
def normalize_text_en(text: str) -> str:
    """تطبيع إنجليزي مبسّط."""
    if not text: return ""
    t = text.lower()
    t = re.sub(r"[-\u2013\u2014/]", " ", t)
    t = re.sub(r"[()]", " ", t)
    t = re.sub(r"[,.;:]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ==============================
# CSV feature helpers
# ==============================
def to_feature_key(col: str) -> str:
    s = (col or "").strip().lower()
    s = s.replace("(", " ").replace(")", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    return s

def load_symptom_keys(csv_path: str) -> List[str]:
    """
    يقرأ الهيدر من grouped_by_department_merged.csv ويستخرج مفاتيح الأعراض.
    """
    if not (csv_path and os.path.exists(csv_path)): return []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f); header = next(rdr, [])
    symptom_cols = [c for c in header if c not in ("Disease", "Department")]
    return list(dict.fromkeys(to_feature_key(c) for c in symptom_cols))

def load_top_features(importances_csv: str, top_n: int = 0) -> List[str]:
    """
    يقرأ feature_importance_rf_et.csv ويعيد أعلى N ميزات.
    إذا كان top_n=0 يعيد قائمة فارغة (لا انتقاء؛ نستخدم كل الأعراض).
    """
    if not ((importances_csv and os.path.exists(importances_csv)) and top_n > 0): return []
    import pandas as pd
    df = pd.read_csv(importances_csv)
    feature_col = df.columns[0]
    df["feature_key"] = df[feature_col].apply(to_feature_key)
    rf_col = [c for c in df.columns if c.lower().startswith("rf")]
    et_col = [c for c in df.columns if c.lower().startswith("et")]
    rf = rf_col[0] if rf_col else None
    et = et_col[0] if et_col else None
    if rf and et:
        df["mean_importance"] = df[[rf, et]].mean(axis=1)
        df = df.sort_values("mean_importance", ascending=False)
        return list(df["feature_key"].head(top_n))
    return list(df["feature_key"].head(top_n))

# ==============================
# English strong synonyms (phrase -> key)
# ==============================
EN_STRONG_SYM = {
    "fever": "fever_general",
    "high temperature": "fever_general",
    "dry cough": "cough_dry_or_chronic",
    "chronic cough": "cough_dry_or_chronic",
    "shortness of breath": "difficulty_breathing",
    "breathlessness": "difficulty_breathing",
    "dyspnea": "difficulty_breathing",
    "wheezing": "wheezing",
    "wheeze": "wheezing",
    "chest pain": "chest_pain",
    "nausea": "nausea_or_vomiting",
    "vomiting": "nausea_or_vomiting",
    "throwing up": "nausea_or_vomiting",
    "fatigue": "fatigue_general",
    "tiredness": "fatigue_general",
    "rash": "rash_general",
    "skin rash": "rash_general",
    "scaly patches on the skin": "scaly_patches_on_the_skin",
    "jaundice": "jaundice",
    "yellowing of skin": "jaundice",
    "yellowing of eyes": "jaundice",
    "swelling": "swelling",
    "edema": "swelling",
    "leg swelling": "swelling_of_the_legs_or_ankles",
    "ankle swelling": "swelling_of_the_legs_or_ankles",
    "blood in urine": "blood_in_urine_or_stool",
    "blood in stool": "blood_in_urine_or_stool",
    "decreased appetite": "decreased_appetite",
    "poor appetite": "decreased_appetite",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "fracture": "fractures",
    "fractures": "fractures",
    "headache": "headache_general",
    "headaches": "headache_general",
    "post nasal drip": "post_nasal_drip",
    "post-nasal drip": "post_nasal_drip",
    "sore throat": "sore_throat",
}

def key_to_phrase(k: str) -> str:
    return (k or '').replace('_', ' ').strip()

def en_phrases_for_key(key: str) -> List[str]:
    phrases = [key_to_phrase(key)]
    for ph, k in EN_STRONG_SYM.items():
        if k == key: phrases.append(ph)
    return list(dict.fromkeys(phrases))

# ==============================
# English pattern builders (flex)
# ==============================
def compile_en_phrase(p: str) -> re.Pattern:
    """مطابقة مرنة للإنجليزية: مسافات/شرطات بين الكلمات + حدود كلمات."""
    tokens = [t for t in re.split(r"\s+", p.lower().strip()) if t]
    if not tokens: return re.compile(r"$^")
    sep = r"(?:\s+|[-–—])"
    escaped = [re.escape(t) for t in tokens]
    body = sep.join(escaped)
    return re.compile(rf"\b{body}\b")

# ==============================
# English negation builders
# ==============================
NEG_PREFIXES = [
    r"no\s+(?:signs|symptoms|complaints|history)\s+of\s+",
    r"not\s+",
    r"without\s+",
    r"denies\s+",
    r"denied\s+",
    r"free\s+of\s+",
    r"no\s+evidence\s+of\s+",
    r"does\s+not\s+have\s+",
    r"has\s+no\s+",
    r"no\s+history\s+of\s+",
]

def compile_en_negation(p: str) -> re.Pattern:
    """يبني نمط نفي إنجليزي مرن قبل العبارة المعطاة."""
    core = compile_en_phrase(p).pattern
    core_body = core[2:-2] if core.startswith(r"\b") and core.endswith(r"\b") else core
    prefix_alt = r"(?:%s)" % "|".join(NEG_PREFIXES)
    patt = rf"{prefix_alt}{core_body}"
    return re.compile(patt)

def build_en_patterns(keys: List[str]) -> List[Tuple[re.Pattern, str]]:
    pats, added = [], set()
    for k in keys:
        for phrase in en_phrases_for_key(k):
            patt = compile_en_phrase(phrase); sig = (patt.pattern, k)
            if sig not in added: pats.append((patt, k)); added.add(sig)
    return pats

def build_en_neg_patterns(keys: List[str]) -> List[Tuple[re.Pattern, str]]:
    pats, added = [], set()
    for k in keys:
        for phrase in en_phrases_for_key(k):
            patt = compile_en_negation(phrase); sig = (patt.pattern, k)
            if sig not in added: pats.append((patt, k)); added.add(sig)
    return pats

# ==============================
# Arabic synonyms (seed; تُدمَج مع الخارجي)
# ==============================
AR_SYNONYMS: Dict[str, List[str]] = {
    # شائعة
    "fever_general": ["حمى", "حرارة", "سخونه", "سخونة", "ارتفاع الحرارة", "حمي", "حراره"],
    "cough": ["سعال", "كحه", "كحة", "يكح", "كح"],
    "cough_dry_or_chronic": ["سعال جاف", "كحه جافه", "كحة جافة", "كحة ناشفة", "سعال ناشف"],
    "difficulty_breathing": ["ضيق نفس", "صعوبه تنفس", "صعوبة تنفس", "ضيق بالتنفس", "نهجان", "كتمه", "كتمة", "اختناق", "تعب بالتنفس"],
    "chest_pain": ["الم صدر", "وجع صدر", "الم في الصدر", "وجع في الصدر", "كتمه صدر", "كتمة صدر"],
    "wheezing": ["صفير", "ازيز", "أزيز", "ربو"],
    "nausea_or_vomiting": ["غثيان", "استفراغ", "تقيؤ", "ترجيع"],
    "fatigue_general": ["ارهاق", "تعب", "خمول"],
    "headache_general": ["صداع", "الم راس", "وجع راس", "الم في الراس", "وجع في الراس"],
    "rash_general": ["طفح جلدي"],
    "scaly_patches_on_the_skin": ["قشور على الجلد", "تقشر الجلد"],
    "jaundice": ["يرقان", "اصفرار الجلد", "اصفرار العين"],
    "swelling": ["تورم", "انتفاخ"],
    "swelling_of_the_legs_or_ankles": ["تورم الساقين", "انتفاخ الساقين", "تورم القدمين", "انتفاخ القدمين"],
    "blood_in_urine_or_stool": ["دم في البول", "دم في البراز", "بول مدمى", "تغوط مع دم"],
    "decreased_appetite": ["فقدان شهية", "قله شهية", "قلة شهية"],
    "loss_of_appetite": ["انعدام شهية", "عدم شهية"],
    "fractures": ["كسور", "كسر"],
    "sore_throat": ["التهاب حلق", "الم حلق", "احتقان حلق"],
    "post_nasal_drip": ["تقاطر انفي", "افرازات خلفية بالأنف", "بلغم خلفي"],
    "sleep_or_appetite_changes": ["تغيرات نوم", "تغيرات شهية"],

    # الإضافات لتغطية العناصر الناقصة
    "seizures": ["نوبات صرع", "تشنجات"],
    "distinct_facial_features_small_jaw": ["ملامح وجه مميزة", "صغر الفك", "فك صغير"],
    "fragile_bones": ["عظام هشة", "هشاشة العظام"],
    "excessive_worrying": ["قلق مفرط", "قلق زائد"],
    "elevated_cholesterol_levels": ["ارتفاع الكوليسترول", "فرط كوليسترول"],
    "difficulty_swallowing": ["صعوبة البلع", "عسر البلع"],
    "restlessness": ["تململ", "أرق"],
    "rough": ["خشونة الجلد", "خشن"],
    "difficulty_bonding_with_the_baby": ["صعوبة الارتباط بالطفل", "ضعف الترابط مع الطفل"],
    "or_face": ["الوجه", "وجه"],
    "sweating": ["تعرق"],
    "emptiness": ["شعور بالفراغ"],
    "usually_in_the_evening_or_at_night": ["غالبًا في المساء أو الليل"],
    "often_in_the_limbs": ["غالبًا في الأطراف"],
    "abnormal_sweating": ["تعرق غير طبيعي", "فرط تعرق"],
    "mood_swings": ["تقلبات المزاج"],
    "irregular_periods": ["عدم انتظام الدورة", "اضطراب الطمث"],
    "usually_in_sun_exposed_areas": ["عادةً في المناطق المعرضة للشمس"],
    "nipple_changes": ["تغيرات الحلمة"],
    "limited_range_of_motion": ["مدى حركة محدود", "تيبس المفصل"],
    "especially_at_night_or_in_response_to_triggers_such_as_allergens_or_exercise": ["خصوصًا ليلاً أو مع محفزات مثل مسببات الحساسية أو التمرين"],
    "loss_of_bladder_or_bowel_control": ["سلس بولي أو برازي", "فقدان السيطرة على البول أو البراز"],
}

# ==============================
# Arabic auto-synonyms generator
# ==============================
TOKEN_MAP: Dict[str, List[str]] = {
    "pain": ["ألم", "وجع"], "ache": ["ألم", "وجع"],
    "abdominal": ["بطن", "بطني"], "back": ["ظهر"], "chest": ["صدر"], "shoulder": ["كتف"],
    "neck": ["رقبة"], "arm": ["ذراع", "يد"], "leg": ["ساق", "قدم"], "hip": ["ورك"], "knee": ["ركبة"], "head": ["راس", "رأس"], "face": ["وجه", "الوجه"],
    "shortness of breath": ["ضيق نفس", "كتمة", "نهجان"], "dyspnea": ["ضيق نفس"], "wheezing": ["صفير", "أزيز"], "cough": ["سعال", "كحة"],
    "dry": ["جاف", "ناشف"], "chronic": ["مزمن"], "productive": ["بلغمي"], "chest congestion": ["احتقان صدر", "كتمة صدر"],
    "sore throat": ["التهاب حلق", "ألم حلق", "احتقان حلق"], "post nasal drip": ["افرازات خلفية بالأنف", "بلغم خلفي", "تقاطر أنفي"],
    "nasal congestion": ["انسداد أنف", "زكمة"], "nasal discharge": ["رشح", "سائل أنفي", "سيلان أنفي"],
    "nausea": ["غثيان"], "vomiting": ["استفراغ", "ترجيع", "تقيؤ"], "diarrhea": ["إسهال"], "constipation": ["إمساك"],
    "bloating": ["نفخة", "انتفاخ البطن"], "blood in stool": ["دم في البراز"], "hematemesis": ["قيء مدمى", "استفراغ دم"],
    "melena": ["براز أسود"], "abdominal swelling": ["تورم البطن", "استسقاء"], "ascites": ["استسقاء"],
    "bowel movement changes": ["تغيرات التبرز"], "bowel irregularities": ["اضطرابات الأمعاء"],
    "burning sensation during urination": ["حرقة بول", "حرقان بول", "ألم أثناء التبول"],
    "painful urination": ["حرقة بول", "حرقان بول", "ألم أثناء التبول"], "dysuria": ["حرقة بول", "حرقان بول"],
    "urinary frequency": ["كثرة التبول", "تكرر التبول"], "blood in urine": ["دم في البول", "بول مدمى"],
    "abnormal urine": ["بول غير طبيعي"], "dark urine": ["بول داكن"], "decreased urine output": ["نقص كمية البول"],
    "increased urination": ["زيادة التبول"], "frequent urination": ["كثرة التبول"], "urgency": ["إلحاح بولي"],
    "rash": ["طفح جلدي"], "itching": ["حكة", "هرش"], "pruritus": ["حكة", "هرش"], "jaundice": ["يرقان", "اصفرار الجلد", "اصفرار العين"],
    "acne": ["حب شباب"], "blurred vision": ["تشوش رؤية", "رؤية ضبابية"], "double vision": ["ازدواجية رؤية"], "vision loss": ["فقدان البصر"],
    "blind spots": ["بقع عمياء"], "changes in skin color or temperature": ["تغيرات لون الجلد أو حرارته"],
    "fever": ["حمى", "ارتفاع حرارة", "سخونة"], "fatigue": ["إرهاق", "تعب", "خمول"], "body aches": ["آلام الجسم"],
    "weight loss": ["نقص وزن"], "weight gain": ["زيادة وزن"], "loss of appetite": ["فقدان شهية", "عدم شهية"],
    "decreased appetite": ["قلة شهية"], "sleep or appetite changes": ["تغيرات النوم", "تغيرات الشهية"],
    "balance issues": ["مشاكل اتزان"], "dizziness": ["دوخة"], "syncope": ["إغماء"], "palpitations": ["خفقان"],
    "bruising": ["كدمات"], "bleeding tendencies": ["ميل للنزف"], "bleeding and bruising": ["نزف وكدمات"], "anxiety": ["قلق"],
    "breast lump": ["كتلة في الثدي"],

    # إضافات
    "seizure": ["نوبة صرع", "تشنج"], "seizures": ["نوبات صرع", "تشنجات"],
    "fragile bones": ["عظام هشة", "هشاشة العظام"],
    "difficulty swallowing": ["صعوبة البلع", "عسر البلع"],
    "restlessness": ["تململ", "أرق"],
    "abnormal sweating": ["تعرق غير طبيعي", "فرط تعرق"],
    "mood swings": ["تقلبات المزاج"],
    "irregular periods": ["عدم انتظام الدورة", "اضطراب الطمث"],
}

EXTRA_BY_KEY: Dict[str, List[str]] = {
    "swelling_of_the_legs_or_ankles": ["تورم الساقين", "تورم الكاحلين", "انتفاخ القدمين"],
    "blood_in_urine_or_stool": ["دم في البول", "دم في البراز"],
    "chest_pain_sharp_or_dull": ["ألم صدر حاد", "ألم صدر خفيف"],
    "abdominal_swelling_ascites": ["تورم البطن", "استسقاء"],
    "changes_in_sleep_and_appetite": ["تغيرات النوم", "تغيرات الشهية"],
    "blurred_or_distorted_central_vision": ["تشوش رؤية", "رؤية ضبابية"],
}

def auto_ar_from_phrase(phrase: str) -> List[str]:
    """يبني مرادفات عربية تلقائيًا من العبارة الإنجليزية اعتمادًا على TOKEN_MAP وبعض التركيبات."""
    p = phrase.lower().strip()
    out: List[str] = []
    if "pain" in p or "ache" in p: out += ["ألم", "وجع"]
    for eng, syns in TOKEN_MAP.items():
        if eng in p: out += syns
    # تركيبات مفيدة
    if "bleeding" in p and "wound" in p: out += ["نزف من الجرح", "نزيف من الجرح"]
    if "ear" in p and "drainage" in p: out += ["إفرازات أذنية", "سائل من الأذن"]
    if ("hearing loss" in p) or ("hearing" in p and ("loss" in p or "muffled" in p)): out += ["فقدان السمع", "ضعف السمع"]
    if "vision" in p and ("loss" in p or "problems" in p or "blurred" in p or "double" in p):
        out += ["مشاكل الرؤية", "فقدان البصر", "تشوش رؤية", "رؤية ضبابية", "ازدواجية رؤية"]
    if "urine" in p and ("blood" in p or "dark" in p or "decreased" in p or "increased" in p or "frequent" in p):
        out += ["دم في البول", "بول داكن", "نقص كمية البول", "زيادة التبول", "كثرة التبول"]
    if "abdomen" in p and ("swollen" in p or "tender" in p): out += ["بطن متورم", "بطن مؤلم"]
    if "salivary glands" in p or "parotid" in p: out += ["تورم الغدد اللعابية", "تورم الغدة النكفية"]
    if "liver" in p and "spleen" in p: out += ["تضخم الكبد أو الطحال"]
    if "ovarian cysts" in p: out += ["أكياس مبيضية"]
    if "raynaud" in p: out += ["مرض رينو", "ظاهرة رينو"]
    # تنظيف + دمج
    norm = []
    for s in out:
        s = re.sub(r"\s+", " ", s).strip()
        if s: norm.append(s)
    return list(dict.fromkeys(norm))

def ensure_ar_synonyms_for_all(keys: List[str]) -> int:
    """
    يضمن وجود مرادفات عربية لكل مفتاح:
    - توليد تلقائي إن لم توجد في AR_SYNONYMS
    - إضافة EXTRA_BY_KEY إن وُجدت
    """
    count = 0
    for k in keys:
        if not AR_SYNONYMS.get(k):
            phrase = key_to_phrase(k)
            syns = auto_ar_from_phrase(phrase) + EXTRA_BY_KEY.get(k, [])
            syns = list(dict.fromkeys(syns))
            if syns:
                AR_SYNONYMS[k] = syns
                count += 1
    return count

# ==============================
# Arabic pattern builders
# ==============================
def _core_pattern(phrase: str) -> re.Pattern:
    """نمط عربي بسيط بعد التطبيع، يستبدل المسافات بـ \\s+."""
    p_norm = normalize_text(phrase)
    core = re.escape(p_norm).replace(r"\ ", r"\s+")
    return re.compile(core)

def _neg_pattern(phrase: str) -> re.Pattern:
    """
    نمط نفي عربي موسّع:
    - (لا|ما) {العبارة}
    - (لا|ما) (يوجد|في) {العبارة}
    - لا وجود {العبارة} / بدون {العبارة} / خالي من {العبارة}
    - ما فيه|ما في {العبارة} / ما عنده|ما عندها|ما عندهم {العبارة}
    - لا يعاني من / لا يشكو من / غير موجود / عدم وجود {العبارة}
    """
    p_norm = normalize_text(phrase)
    core = re.escape(p_norm).replace(r"\ ", r"\s+")
    alts = [
        rf"(?:لا|ما)\s+{core}",
        rf"(?:لا|ما)\s+(?:يوجد|في)\s+{core}",
        rf"لا\s+وجود\s+{core}",
        rf"بدون\s+{core}",
        rf"خالي\s+من\s+{core}",
        rf"(?:ما\s+فيه|ما\s+في)\s+{core}",
        rf"(?:ما\s+عنده|ما\s+عندها|ما\s+عندهم)\s+{core}",
        rf"لا\s+يعاني\s+من\s+{core}",
        rf"لا\s+يشكو\s+من\s+{core}",
        rf"غير\s+موجود\s+{core}",
        rf"عدم\s+وجود\s+{core}",
    ]
    return re.compile("|".join(alts))

# ==============================
# Load extra Arabic synonyms from CSV (optional; افتراضيًا ar_synonyms_medical.csv)
# ==============================
def _load_ar_synonyms_from_csv(path: str) -> Dict[str, List[str]]:
    """
    تحميل مرادفات عربية من CSV:
    - الأعمدة: feature_key, arabic_synonyms [, english_phrase]
    - المرادفات مفصولة بـ (،) أو (;) أو (,) أو سطر جديد
    """
    syn: Dict[str, List[str]] = {}
    if not (path and os.path.exists(path)): return syn
    import csv as _csv
    with open(path, "r", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        for row in rdr:
            k = (row.get("feature_key") or "").strip().lower()
            s = (row.get("arabic_synonyms") or "").strip()
            if not k or not s: continue
            parts = re.split(r"[\u060c;,\\\n/]+", s)
            phrases = [normalize_text(p) for p in parts if p.strip()]
            if phrases: syn[k] = list(dict.fromkeys(phrases))
    return syn

# ==============================
# NLPAgent
# ==============================
class NLPAgent:
    def __init__(
        self,
        m1_keys: List[str],
        symptoms_csv: str = "grouped_by_department_merged.csv",
        importances_csv: str = "feature_importance_rf_et.csv",
        top_n: int = 0,                              # ← افتراضيًا يشمل كل الأعراض
        ar_synonyms_csv: Optional[str] = "ar_synonyms_medical.csv"  # ← يحمل الملف الطبي تلقائيًا
    ):
        """
        m1_keys: قائمة المفاتيح المقبولة.
        ar_synonyms_csv: مسار CSV لمرادفات عربية خارجية (افتراضيًا ar_synonyms_medical.csv).
        """
        self.m1_keys = set((k or "").lower() for k in (m1_keys or []))

        # تحميل مفاتيح الأعراض
        all_symptom_keys = load_symptom_keys(symptoms_csv) or []
        if not all_symptom_keys:
            # احتياطي إن لم يتوفر CSV
            all_symptom_keys = [
                "fever_general","cough","cough_dry_or_chronic","difficulty_breathing",
                "chest_pain","wheezing","nausea_or_vomiting","fatigue_general",
                "rash_general","scaly_patches_on_the_skin","jaundice","swelling",
                "swelling_of_the_legs_or_ankles","blood_in_urine_or_stool",
                "decreased_appetite","loss_of_appetite","fractures",
                "headache_general","sore_throat","post_nasal_drip","sleep_or_appetite_changes",
                "seizures","difficulty_swallowing","mood_swings","irregular_periods",
                "limited_range_of_motion","sweating","abnormal_sweating",
            ]

        top_keys = load_top_features(importances_csv, top_n=top_n)  # [] إذا top_n=0
        target_keys = set(k for k in all_symptom_keys if (not top_keys) or (k in top_keys)) or set(all_symptom_keys)
        self._returnable_keys = set(k for k in target_keys if k in self.m1_keys)

        # دمج مرادفات عربية خارجية (إن وُجدت)
        if ar_synonyms_csv:
            extra = _load_ar_synonyms_from_csv(ar_synonyms_csv)
            for k, phrases in extra.items():
                if k in AR_SYNONYMS:
                    AR_SYNONYMS[k] = list(dict.fromkeys(AR_SYNONYMS[k] + phrases))
                else:
                    AR_SYNONYMS[k] = phrases

        # توليد مرادفات عربية تلقائيًا لباقي المفاتيح
        self._ar_auto_count = ensure_ar_synonyms_for_all(list(self._returnable_keys))

        # بناء الأنماط
        self._ar_pos: List[Tuple[re.Pattern, str]] = []
        self._ar_neg: List[Tuple[re.Pattern, str]] = []
        for k in self._returnable_keys:
            for ph in AR_SYNONYMS.get(k, []):
                self._ar_pos.append((_core_pattern(ph), k))
                self._ar_neg.append((_neg_pattern(ph), k))

        self._en_pos = build_en_patterns(list(self._returnable_keys))
        self._en_neg = build_en_neg_patterns(list(self._returnable_keys))

    def extract(self, raw_text: str) -> Dict[str, int]:
        """
        استخراج الأعراض الإيجابية (0/1) من نص عربي/إنجليزي
        مع قمع النفي قبل التثبيت.
        """
        t_ar = normalize_text(raw_text or "")
        t_en = normalize_text_en(raw_text or "")
        findings: Dict[str, int] = {}

        # 1) رصد النفي
        negated = set()
        for patt, key in self._ar_neg:
            try:
                if patt.search(t_ar) and key in self._returnable_keys:
                    negated.add(key)
            except re.error:
                continue
        for patt, key in self._en_neg:
            try:
                if patt.search(t_en) and key in self._returnable_keys:
                    negated.add(key)
            except re.error:
                continue

        # 2) العربية
        for patt, key in self._ar_pos:
            if key in negated: continue
            try:
                if patt.search(t_ar) and key in self._returnable_keys:
                    findings[key] = 1
            except re.error:
                continue

        # 3) الإنجليزية
        for patt, key in self._en_pos:
            if key in negated: continue
            try:
                if patt.search(t_en) and key in self._returnable_keys:
                    findings[key] = 1
            except re.error:
                continue

        return findings

    def extract_negations(self, raw_text: str, candidate_keys: List[str]) -> Dict[str, int]:
        """
        واجهة توافقية قديمة — النفي مُعالج داخل extract.
        تُعيد أمثلة بسيطة عند الطلب.
        """
        t = normalize_text(raw_text or "")
        neg = {}
        if "cough" in candidate_keys and re.search(r"(?:لا|ما)\s+(?:يوجد|في)?\s+(?:سعال|كحه)", t): neg["cough"] = 0
        if "fever_general" in candidate_keys and re.search(r"(?:لا|ما)\s+(?:يوجد|في)?\s+(?:حمى|حراره|حرارة|سخونه|سخونة)", t): neg["fever_general"] = 0
        if "chest_pain" in candidate_keys and re.search(r"(?:لا|ما)\s+(?:يوجد|في)?\s+(?:الم|وجع)\s+(?:صدر|الصدر)", t): neg["chest_pain"] = 0
        if "difficulty_breathing" in candidate_keys and re.search(r"(?:لا|ما)\s+(?:يوجد|في)?\s+(?:ضيق\s+نفس|صعوبه\s+تنفس|صعوبة\s+تنفس|نهجان|كتمه|كتمة|اختناق)", t): neg["difficulty_breathing"] = 0
        return {k: v for k, v in neg.items() if k in self._returnable_keys}


# ==============================
# مثال تشغيل اختياري
# ==============================
if __name__ == "__main__":
    m1_keys = [
        "fever_general","cough","difficulty_breathing","chest_pain","wheezing",
        "nausea_or_vomiting","fatigue_general","rash_general","jaundice","swelling",
        "swelling_of_the_legs_or_ankles","blood_in_urine_or_stool","decreased_appetite",
        "loss_of_appetite","fractures","headache_general","sore_throat","post_nasal_drip",
        "seizures","difficulty_swallowing","mood_swings","irregular_periods",
        "limited_range_of_motion","sweating","abnormal_sweating",
    ]
    # يقرأ ar_synonyms_medical.csv تلقائيًا (إن وُجد)
    agent = NLPAgent(m1_keys=m1_keys, top_n=0)
    txt = "المريض لا يعاني من سعال لكن لديه ضيق نفس وتقلبات المزاج."
    print(agent.extract(txt))
