
# controller.py — تدفّق: نصّ حر → (واجهة: هل لديك أعراض أخرى؟) → نعم/لا → Checkbox عند الحاجة → M2
# -*- coding: utf-8 -*-
import os
from functools import wraps
from flask import Flask, request, jsonify, make_response, send_from_directory, Response
from flask_cors import CORS


# ===== طبقة اللغة & OpenRouter (اختياري) =====
from llm_utils import (
    llm_normalize_user_text,
    llm_extract_symptoms,
    enhance_and_localize_response,
    nlg_format_result_summary,
    DEFAULT_UI_LANG
)

# ===== المنطق الطبي الأساسي (B/M1/M2) =====
from core import (
    m1_feature_pool,
    canonicalize_to_m1_keys,
    m1_is_sufficient,
    run_M1_on_answers,
    run_M2,
    core_handle_answers,
    DEPT_TO_DISEASE,
    get_required_keys_for_disease,
    generate_questions,
    generate_questions_from_columns,
    normalize_value_for_key,
    m1_followups,
    BINARY_KEYS, AUTO_MAPPING
)





# === asked helpers: always store list in sess; cast to set when editing ===

def _normalize_key(x):
    if x is None:
        return None
    try:
        s = str(x).strip().lower()
        return s if s else None
    except Exception:
        return None

def _extract_keys_any(obj):
    """
    يحوّل أي تمثيل محتمل لـ asked إلى قائمة مفاتيح نصية موحدة:
    - set/list/tuple من سلاسل
    - dict => نأخذ المفاتيح
    - list of dicts => نجرب ('value' ثم 'name' ثم 'key')
    """
    out, seen = [], set()
    if isinstance(obj, (set, list, tuple)):
        it = obj
    elif isinstance(obj, dict):
        it = obj.keys()
    elif obj is None:
        it = []
    else:
        it = [obj]

    for item in it:
        if isinstance(item, dict):
            for cand in ("value", "name", "key"):
                if cand in item:
                    nk = _normalize_key(item[cand])
                    if nk and nk not in seen:
                        seen.add(nk); out.append(nk)
                    break
        else:
            nk = _normalize_key(item)
            if nk and nk not in seen:
                seen.add(nk); out.append(nk)
    return out

def _get_asked_set(sess):
    """ارجع asked كـ set موحّد (lower+strip) من أي تمثيل مخزّن."""
    raw = sess.get("asked")
    keys = _extract_keys_any(raw)
    return set(keys)

def _store_asked(sess, asked_set):
    """
    خزّن asked كـ list JSON-friendly مع الحفاظ على الترتيب القديم
    وإضافة العناصر الجديدة في النهاية.
    """
    old = _extract_keys_any(sess.get("asked"))
    old_seen = set(old)
    merged = old[:]
    for k in asked_set or set():
        nk = _normalize_key(k)
        if nk and nk not in old_seen:
            old_seen.add(nk)
            merged.append(nk)
    sess["asked"] = merged

# ===== Helpers: تعليم الأعراض المعروضة كـ asked + استخراج قيم الاختيارات =====
def _extract_symptom_values_from_ask(ask_list):
    """يستخرج القيم (value) من أسئلة symptoms_m1 في قائمة ask."""
    items = ask_list if isinstance(ask_list, list) else [ask_list]
    vals = []
    for q in items:
        if isinstance(q, dict) and q.get("name") == "symptoms_m1":
            for opt in (q.get("options") or []):
                v = opt.get("value") if isinstance(opt, dict) else opt
                if v is not None:
                    vals.append(str(v).strip())
    return vals


def _mark_presented_as_asked(sess, ask_list):
    """
    يعلّم كل الأعراض التي عُرضت (في symptoms_m1) كـ asked حتى لو لم يختَر المستخدم شيئًا،
    حتى لا تُعاد نفس الدفعة في الجولات التالية.
    """
    presented = _extract_symptom_values_from_ask(ask_list)
    if not presented:
        return

    asked = _get_asked_set(sess)
    asked.update({p for p in presented})
    _store_asked(sess, asked)
    sess["asked_symptoms"] = list(asked)

    # ✅ تخزين الدفعة الحالية لاستخدامها لاحقًا في receive_answers_for_sid لملء 0
    sess["presented_current_batch"] = [str(p).strip() for p in presented if p is not None]



# ===== NLPAgent (اعتمادية اختيارية) =====
USE_NLP_AGENT = os.getenv("USE_NLP_AGENT", "1") == "1"
try:
    from agents_nlp import NLPAgent
except Exception as e:
    NLPAgent = None
    print(f"[NLPAgent][IMPORT][WARN] {e} — proceeding without NLPAgent")

nlp_agent = NLPAgent(m1_keys=m1_feature_pool()) if (USE_NLP_AGENT and NLPAgent) else None
BASE_DIR = os.path.dirname(__file__)
SYMPTOMS_CSV    = os.path.join(BASE_DIR, "grouped_by_department_merged.csv")
IMPORTANCES_CSV = os.path.join(BASE_DIR, "feature_importance_rf_et.csv")
AR_CSV          = os.getenv("AR_SYNONYMS_CSV", os.path.join(BASE_DIR, "ar_synonyms_medical.csv"))

TOP_N = int(os.getenv("TOP_N", "0"))
TOP_FIRST_QUESTIONS = int(os.getenv("TOP_FIRST_QUESTIONS", "5"))  # للاحتفاظ به للاتساق

# ==============================
# مزيّن يضمن أن الاستجابة JSON كائن وليست null
# ==============================

def _sanitize_m1_checkbox(q):
    """
    يُبقي فقط الخيارات التي قيمتها ضمن أعمدة PKL (m1_feature_pool).
    يعمل فقط على السؤال: symptoms_m1
    """
    try:
        if isinstance(q, dict) and q.get("name") == "symptoms_m1":
            pool = {(k or "").lower() for k in (m1_feature_pool() or [])}
            filtered = []
            for opt in (q.get("options") or []):
                # opt قد يكون dict أو نص
                if isinstance(opt, dict):
                    v = opt.get("value", opt.get("label"))
                else:
                    v = opt
                if str(v).lower() in pool:
                    filtered.append(opt)
            q["options"] = filtered
    except Exception as e:
        print("[SANITIZE][WARN]", e)
    return q

def _sanitize_ask_list(ask):
    """
    يطبق الفلتر على قائمة/عنصر ask، ويُعيد قائمة جاهزة للإرسال.
    لا يغيّر أسئلة M2 أو اللغة أو النص الحر؛ يُعدّل symptoms_m1 فقط.
    """
    items = ask if isinstance(ask, list) else [ask]
    out = []
    for item in items:
        out.append(_sanitize_m1_checkbox(item))
    return out

def ensure_json_object(route_func):
    """
    يضمن أن الاستجابة ليست null:
    - إذا رجع None -> {"ok": True}
    - إذا رجع نص/قائمة/رقم -> نلفّه داخل {"data": ...}
    - نضيف sid إن أمكن
    - إذا رجع Flask Response جاهز (make_response / jsonify) نعيده كما هو
    """
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        resp = route_func(*args, **kwargs)

        # إذا كانت الاستجابة Response جاهزة، اتركها كما هي
        if isinstance(resp, Response):
            return resp

        data = None
        rest = None
        if isinstance(resp, tuple) and len(resp) >= 1:
            data = resp[0]
            rest = resp[1:]
        else:
            data = resp

        # لا تسمح بالـ null
        if data is None:
            data = {"ok": True}

        # إن لم تكن dict، لفّها داخل كائن
        if not isinstance(data, dict):
            data = {"data": data, "ok": True}

        # أضف sid إن أمكن ولم يكن موجودًا
        sid = request.cookies.get("sid") or request.headers.get("X-Session-Id") or request.args.get("sid")
        if sid and "sid" not in data:
            data["sid"] = sid

        if rest:
            return jsonify(data), *rest
        return jsonify(data)
    return wrapper

# ==============================
# جلسات المستخدم موحَّدة بـ sid
# ==============================
SESSIONS = {}  # sid -> حالة الجلسة


def get_sid(req, allow_create=True):
    from uuid import uuid4

    json_sid = None
    if req.is_json:
        json_sid = (req.get_json(silent=True) or {}).get("sid")

    sid = (
        req.cookies.get("sid")
        or req.headers.get("X-Session-Id")
        or req.args.get("sid")
        or json_sid
    )

    if not sid:
        if not allow_create:
            return None
        sid = str(uuid4())


    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "answers": {},
            "answers_accumulated": {},   # جديد: لتجميع إجابات الـ checkbox
            "asked_symptoms": [],        # جديد: قائمة الأعراض التي سُئلت
            "pending_questions": [],
            "mode": "M1",
            "target_disease": None,
            "asked": set(),
            "lang": None,
            "last_user_text": "",
            "stage": "init",
            "skip_count": 0
        }

    return sid

app = Flask(__name__)
@app.route("/health", methods=["GET"])
def health():
    from core import health_check
    return jsonify(health_check())

def normalize_yes_no(s):
    if s is None: return None
    s = str(s).strip().lower()
    yes = {"yes","y","نعم","ايه","أيوه","تمام","sure","ok","اه"}
    no  = {"no","n","لا","كلا","مش","nope","not","مافي"}
    if s in yes: return "yes"
    if s in no:  return "no"
    return None

def normalize_intent(intent, message=None):
    if intent in ("more_yes","more_no","symptoms",""):
        return intent or ("symptoms" if message else "")
    yn = normalize_yes_no(intent)
    if yn == "yes": return "more_yes"
    if yn == "no":  return "more_no"
    return "symptoms" if message else ""

# ===== عناصر أسئلة الواجهة =====

def build_free_text_question(initial: bool = False, lang: str = "ar"):
    q_ar = "اذكر لي كل الأعراض التي تشعر بها." if initial else "اكتب أي أعراض إضافية (إن وجدت)."
    q_en = "Tell me all the symptoms you have." if initial else "Write any additional symptoms (if any)."
    return {
        "name": "free_text_symptoms",
        "type": "text",
        "q": q_en if lang == "en" else q_ar,  # ✅ اضبط النص حسب اللغة
        "q_ar": q_ar,
        "q_en": q_en
    }

def build_more_symptoms_yesno(lang: str = "ar"):
    q_ar = "هل لديك أعراض أخرى؟"
    q_en = "Do you have any other symptoms?"
    options = [
        {"value": "yes", "label": "Yes" if lang == "en" else "نعم", "label_en": "Yes", "label_ar": "نعم"},
        {"value": "no",  "label": "No"  if lang == "en" else "لا",   "label_en": "No",  "label_ar": "لا"},
    ]
    return {
        "name": "more_symptoms",
        "type": "radio",
        "options": options,
        "q": q_en if lang == "en" else q_ar,  # ✅ اضبط النص حسب اللغة
        "q_ar": q_ar,
        "q_en": q_en
    }

def build_language_question():
    return {
        "name": "user_language",
        "q": "اختر لغتك / Choose your language",
        "type": "radio",
        "options": [
            {"label": "العربية", "value": "ar"},
            {"label": "English", "value": "en"}
        ]
    }


# ==============================
# إعداد Flask/CORS
# ==============================
CORS(app)  # للإنتاج: CORS(app, resources={r"/api/*": {"origins": ["https://YOUR-UI.example.com"]}})

# ==============================
# ربط "/" بشكل صحيح
# ==============================
@app.route("/")
def index():
    return send_from_directory(r"C:\xampp\htdocs\SmartClinic\proj", "index.html")




def _build_answers_for_stop(sess):
    """
    يبني قاموس إجابات للتوقف/التقييم:
    - يبدأ من إجابات المستخدم الفعلية (selected)
    - يُضيف 0 لكل عرض تمّت رؤيته (asked) ولم يُختَر
    - يُرجع القيم بأسماء أعمدة PKL (كما تتوقعها M1)
    """
    pool = m1_feature_pool() or []
    key_map = {k.lower(): k for k in pool}

    # 1) إجابات المستخدم الفعلية بصيغة PKL
    user_ans = sess.get("answers", {}) or {}
    user_ans_canon = canonicalize_to_m1_keys(user_ans, pool) or {}

    # 2) 0 لكل عرض عُرض ولم يُختر
    asked = set(sess.get("asked", set()) or [])
    for a in asked:
        orig = key_map.get(str(a).lower())
        if orig and orig not in user_ans_canon:
            user_ans_canon[orig] = 0

    return user_ans_canon

# ==============================
# /api/controller — مسار إجابات منظمة (Checkbox/حقول M2)
# ==============================
@app.post("/api/controller")
@ensure_json_object
def controller():
    sid = get_sid(request, allow_create=True)


    if not sid:
        return {
            "ok": False,
            "error": "sid مفقود",
            "ui_hint": {"require_sid": True}
        }, 400

    sess = SESSIONS[sid]
    user_input_raw = request.json.get("answers") if request.is_json else None

    if user_input_raw:
        # توحيد + دمج
        try:
            canon = canonicalize_to_m1_keys(user_input_raw, m1_feature_pool())
        except Exception as e:
            print(f"[CONTROLLER][CANON][EXC] {e}")
            canon = {}
        merged = dict(user_input_raw or {})
        merged.update(canon or {})
        receive_answers_for_sid(sid, merged)

        # >>> [NEW] تعامل خاص مع دفعة الـ checkbox عبر core_handle_answers

        if isinstance(user_input_raw, dict):
            checkbox_selected = user_input_raw.get("symptoms_m1") or user_input_raw.get("symptoms_checkbox")

            # ❶ حالة: المستخدم لم يختر أي عرض (قائمة فارغة)
            if isinstance(checkbox_selected, list) and not checkbox_selected:
                # 1) زد عداد التخطي
                sess["skip_count"] = int(sess.get("skip_count", 0)) + 1

                # 2) اطلب دفعة جديدة مع تمرير answered_map لمنع تكرار الأعراض المكتوبة نصًا
                out = generate_questions_from_columns(
                    columns=m1_feature_pool(),
                    asked_set=set(sess.get("asked", set())),
                    batch_size=5,
                    m1_scores=sess.get("M1_scores"),
                    answered_map=sess.get("answers", {}),
                    add_entire_batch_to_asked=False,
                    lang=sess.get("lang", "ar")  # ✅
                )
                if isinstance(out, tuple):
                    q_list, asked_updated = out
                    _store_asked(sess, set(asked_updated or []))
                else:
                    q_list = out

                if q_list:
                    ask_out = q_list if isinstance(q_list, list) else [q_list]
                    ask_out = _sanitize_ask_list(ask_out)
                    _mark_presented_as_asked(sess, ask_out)
                    return enhance_and_localize_response({"sid": sid, "ask": ask_out}, sess.get("lang", "ar"))

                # 3) لا توجد دفعات جديدة: تحقّق من الكفاية/رسالة إرشاد
                answers_for_stop = _build_answers_for_stop(sess)
                sufficient, _ = m1_is_sufficient(answers_for_stop)
                dist = run_M1_on_answers(answers_for_stop) or {}
                if not sufficient:
                    if sess["skip_count"] >= 3:
                        payload = {
                            "sid": sid,
                            "ask": [build_free_text_question(initial=False, lang=sess.get("lang", "ar"))],
                            "note": "لا يمكن التقدم دون اختيار أي أعراض. اكتب أعراضك نصًا أو اختر من القوائم."
                        }
                        return enhance_and_localize_response(payload, sess.get("lang", "ar"))
                    return enhance_and_localize_response(
                        {"sid": sid, "note": "اختر عرضًا واحدًا على الأقل أو اكتب أعراضك نصًا."},
                        sess.get("lang", "ar")
                    )

            # ❷ حالة: المستخدم اختار أعراضًا (قائمة غير فارغة)
            if isinstance(checkbox_selected, list) and checkbox_selected:
                payload = {
                    "sid": sid,                                      
                    "session": {
                        "asked_symptoms": list(sess.get("asked_symptoms", [])),
                        "answers_accumulated": sess.get("answers_accumulated", {}),
                        "M1_scores": sess.get("M1_scores"),
                        "answers": dict(sess.get("answers", {})),
                        "lang": sess.get("lang", "ar")  # ✅ أضف اللغة هنا
                    },

                    "answers": {"symptoms_m1": checkbox_selected}
                }

                # شغّل core ثم طبّق نفس منطقك السابق
                core_resp = core_handle_answers(payload)

                # تحديث الجلسة بما أعاده core
                core_session = core_resp.get("session", {}) or {}
                sess["answers_accumulated"] = core_session.get("answers_accumulated", {})
                sess["asked_symptoms"]      = core_session.get("asked_symptoms", [])
                sess["M1_scores"]           = core_session.get("M1_scores", {})
                _store_asked(sess, set(sess.get("asked_symptoms", [])))

                # إن وُجدت أسئلة جديدة (دفعة تالية)
                if core_resp.get("ask"):
                    ask_list = _sanitize_ask_list(core_resp["ask"])
                    _mark_presented_as_asked(sess, ask_list)
                    return enhance_and_localize_response({"sid": sid, "ask": ask_list}, sess.get("lang", "ar"))

                # لا توجد أسئلة جديدة → استخدم درجات M1 للتقدّم (قد تنتقل إلى M2)
                dist = (core_resp.get("result") or {}).get("M1", {}) or {}
                if dist:
                    answers_for_stop = _build_answers_for_stop(sess)
                    sufficient, _ = m1_is_sufficient(answers_for_stop)
                    dist = run_M1_on_answers(answers_for_stop) or {}
                    if sufficient:
                        top_dept       = max(dist, key=dist.get)
                        mapped_disease = DEPT_TO_DISEASE.get(top_dept, top_dept)
                        sess["mode"]   = "M2"
                        sess["target_disease"] = mapped_disease
                        _ = get_required_keys_for_disease(mapped_disease)
                        q = generate_questions("M2", disease_required=[mapped_disease], lang=sess.get("lang", "ar"))
                        # (إن رغبت، أعد نفس الـ payload الذي استخدمته سابقًا)

                    # إن لم تكن كافية وليس لدى core أسئلة إضافية، سيكمل التدفق الحالي أدناه


    # ----- تدفق السيناريو -----

    if sess["mode"] == "M1":
        # 1) تشغيل/تخزين درجات M1
        answers_for_stop = _build_answers_for_stop(sess)  # يبني جميع الإجابات + 0 لكل سؤال تم عرضه ولم يُختَر
        dist = run_M1_on_answers(answers_for_stop) or {}
        sess["M1_scores"] = dist
        # 2) أسئلة متابعة موجهة للقسم الأعلى

        q = m1_followups(
            sess["answers"],
            top_dept=max(dist, key=dist.get) if dist else None,
            limit=5,
            lang=sess.get("lang", "ar"),          # ✅ اللغة من الجلسة
            m1_scores=sess.get("M1_scores")       # ✅ إن وجدت درجات، مرّرها
        )

        # 3) إن لم تُرجع m1_followups أسئلة، نستخدم generate_questions_from_columns
        if not q:

            q, _ = generate_questions_from_columns(
                columns=m1_feature_pool(),
                asked_set=set(sess["asked"]),
                batch_size=5,
                m1_scores=sess.get("M1_scores"),
                answered_map=sess.get("answers", {}),
                add_entire_batch_to_asked=False,
                lang=sess.get("lang", "ar")  # ✅
            )

        # 4) انتقال إلى M2 إن كانت الأعراض كافية ولم يعد هناك أسئلة
        sufficient, _ = m1_is_sufficient(answers_for_stop)
        if sufficient:
            top_dept = max(dist, key=dist.get) if dist else None
            mapped_disease = DEPT_TO_DISEASE.get(top_dept, top_dept) if top_dept else "HeartDisease"
            sess["mode"] = "M2"
            sess["target_disease"] = mapped_disease
            reqs = get_required_keys_for_disease(mapped_disease)
            q = generate_questions("M2", disease_required=[mapped_disease], lang=sess.get("lang", "ar"))
        # 5) الاستمرار في M1 وإعادة الأسئلة (قائمة دائمًا)


        if q:
            if isinstance(q, dict):
                q = [q]
            # فلترة PKL + تعليم المعروض كـ asked
            q = _sanitize_ask_list(q)
            _mark_presented_as_asked(sess, q)
            return enhance_and_localize_response({"sid": sid, "ask": q}, sess.get("lang", "ar"))

    # ----- وضع M2 -----
    if sess["mode"] == "M2":


        # =========================
        # PATCH: M2 missing fields
        # =========================

        # احسب الحقول الناقصة (names) من required_keys للمرض الحالي
        missing = [
            k for k in get_required_keys_for_disease(sess["target_disease"])
            if k.lower() not in {kk.lower() for kk in sess["answers"].keys()}
        ]

        if missing:
            # [REMOVE] كانت تُستدعى دالة M1 وبـ lang (هذا يسبب TypeError)
            # q = generate_questions_from_columns(
            #     missing,
            #     asked_set=sess["asked"],
            #     lang=sess.get("lang","ar")
            # )

            # [ADD] ✅ توليد أسئلة M2 الصحيحة بالاسم الفعلي للمرض + اللغة
            all_q = generate_questions(
                "M2",
                disease_required=[sess["target_disease"]],
                lang=sess.get("lang", "ar")
            )
            # فلترة الأسئلة لاختيار فقط الحقول الناقصة
            q = [qq for qq in all_q if qq.get("name") in missing]

            # تأكد أن q قائمة دائمًا
            sess["pending_questions"] = q
            return enhance_and_localize_response({"sid": sid, "ask": q if isinstance(q, list) else [q]}, sess.get("lang", "ar"))
        # تشغيل M2 النهائي (بدون تغيير)

        disease = sess["target_disease"]
        reqs = get_required_keys_for_disease(disease)

        # ابنِ دخلًا نظيفًا: أي قيمة فارغة/غير رقمية ← 0
        m2_input = {}
        for k in reqs:
            v = sess["answers"].get(k, 0)
            if v in (None, "", " ", "null"):
                v = 0
            v = normalize_value_for_key(k, v)  # يطبّق خرائط العربية/الإنجليزية للأصناف
            if isinstance(v, str):
                try:
                    v = float(v.strip())
                except Exception:
                    v = 0
            m2_input[k] = v

        final_score = run_M2(disease, data=m2_input)
        if final_score is None:
            return enhance_and_localize_response(
                {"error": "تعذّر حساب النتيجة الآن. جرّب لاحقًا.", "sid": sid},
                sess["lang"]
            )
        # استخراج أعلى قسم من M1 إن وُجد
        m1_scores = sess.get("M1_scores") or {}
        top_department = max(m1_scores, key=m1_scores.get) if m1_scores else None

        final_report = {
            "top_department": top_department,
            "disease": sess["target_disease"],
            "percentage": round(final_score * 100.0, 2)
        }
        # ... بقية الكود كما هو

        try:
            pretty = nlg_format_result_summary(
                final_report["disease"],
                final_report["percentage"],
                target_lang=sess["lang"]
            )
            payload = {"final_diagnosis": final_report, "final_diagnosis_nlg": pretty,"top_department": top_department, "sid": sid}
        except Exception as e:
            print(f"[NLG-RES][ERR] {e}")
            payload = {"final_diagnosis": final_report, "sid": sid}

        # تصفير جلسة هذا المستخدم فقط مع الاحتفاظ باللغة (موحّدة مع get_sid)
        sess_lang = sess["lang"]
        SESSIONS[sid] = {
            "answers": {},
            "answers_accumulated": {},   # مهم للدفعات القادمة
            "asked_symptoms": [],        # مواءمة مع core_handle_answers
            "pending_questions": [],
            "mode": "M1",
            "target_disease": None,
            "asked": set,
            "lang": sess_lang,
            "last_user_text": "",
            "stage": "init"              # إبقاء الحالة الابتدائية
        }
        return enhance_and_localize_response(payload, sess_lang)

    # افتراضي: بدء جلسة جديدة أو تلميح للواجهة
    payload = {"sid": sid, "ui_hint": {"start_with_free_text": True}}
    return enhance_and_localize_response(payload, SESSIONS[sid]["lang"])



# ==============================
# /api/chat_m1 — دالة الحوار النصّي (مطابقة للسيناريو، سؤال "أعراض أخرى" واجهة فقط)
# ==============================


def receive_answers_for_sid(sid, user_answers: dict):
    """
    استقبال الإجابات بشكل غير حساس لحالة الأحرف + تطبيع دائم
    مع توسيع إجابة Checkbox متعددة الأعراض إلى مفاتيح M1 فردية.
    """
    sess = SESSIONS[sid]

    # --- توسيع إجابة الشيك بوكس إلى مفاتيح فردية ---
    expanded = dict(user_answers or {})
    sel = expanded.pop("symptoms_m1", None) or expanded.pop("symptoms_checkbox", None)
    if isinstance(sel, list):
        # اعتبر كل خيار مختار = "yes" (موجب)
        for key in sel:
            if key not in expanded:
                expanded[key] = "yes"

    user_answers = expanded

    # ✅ جديد: ملء 0 تلقائيًا للخيارات المعروضة في الدفعة الحالية وغير المختارة
    presented_batch = [str(x).strip() for x in (sess.get("presented_current_batch") or [])]
    selected_lc = {str(k).strip().lower() for k in (sel or [])}
    missing_in_batch = [x for x in presented_batch if x and x.lower() not in selected_lc]

    if missing_in_batch:
        # ابنِ خريطة صفريّات وحوّلها إلى صيغة أعمدة PKL (canonical)
        zero_map_raw = {k: 0 for k in missing_in_batch}
        zero_canon = canonicalize_to_m1_keys(zero_map_raw, m1_feature_pool() or [])
        for orig_k, _ in (zero_canon or {}).items():
            kl = str(orig_k).strip().lower()
            sess["answers"][kl] = 0  # خزّن 0 في الإجابات
            asked_set = _get_asked_set(sess)
            asked_set.add(kl)
            _store_asked(sess, asked_set)

        # نظّف التخزين المؤقت للدفعة الحالية كي لا يُعاد ملء 0 مرة أخرى
        sess["presented_current_batch"] = []

    # بناء قائمة المفاتيح المسموح بها حسب الوضع الحالي

    if sess["mode"] == "M1":
        allowed = (
            {k.lower() for k in (m1_feature_pool() or [])}
            | {k.lower() for k in BINARY_KEYS}
        )
    else:
        allowed = {
            k.lower() for k in get_required_keys_for_disease(sess.get("target_disease") or "")
        }

    accepted, rejected = 0, 0
    for k, v in (user_answers or {}).items():
        kl = (k or "").lower()
        if kl not in allowed:
            rejected += 1
            continue

        nv = normalize_value_for_key(kl, v)  # التطبيع دائمًا
        sess["answers"][kl] = nv

        asked_set = _get_asked_set(sess)
        asked_set.add(kl)
        _store_asked(sess, asked_set)
        accepted += 1

    if rejected:
        print(f"[RECEIVE][WARN] Rejected {rejected} keys (not in allowed). Accepted={accepted}")



def apply_core_checkbox_batch(sid: str, selected_list: list, sess: dict):
    """
    يشغّل core_handle_answers لدفعة أعراض (checkbox)، ويُحدّث الجلسة،
    ثم يُعيد (ask_next, dist), حيث:
      - ask_next: قائمة أسئلة جديدة إن وُجدت (أو None)
      - dist: توزيع أقسام M1 إن لم توجد أسئلة جديدة (أو {})
    """
    if not isinstance(selected_list, list) or not selected_list:
        return None, {}

    payload = {
        "sid": sid,
        "session": {
            "asked_symptoms": list(sess.get("asked_symptoms", [])),
            "answers_accumulated": sess.get("answers_accumulated", {}),
            "M1_scores": sess.get("M1_scores"),
            "lang": sess.get("lang", "ar"),  # ✅ يتيح للـ core توليد q/label بنفس اللغة
            "answers": dict(sess.get("answers", {}))
        },
        # core_handle_answers يتوقع dict: المفتاح -> قائمة
        "answers": {"symptoms_m1": selected_list}
    }

    core_resp = core_handle_answers(payload)  # من core.py
    core_session = core_resp.get("session", {}) or {}

    # تحديث الجلسة من مخرجات core
    sess["answers_accumulated"] = core_session.get("answers_accumulated", {})
    sess["asked_symptoms"] = core_session.get("asked_symptoms", [])
    sess["M1_scores"] = core_session.get("M1_scores", {})
    _store_asked(sess, set(sess.get("asked_symptoms", [])))  # مواءمة


    if core_resp.get("ask"):
        ask_list = core_resp["ask"]
        # ضمننة القائمة + فلترة M1 من PKL
        ask_list = _sanitize_ask_list(ask_list)
        # تعليم الأعراض المعروضة كـ asked حتى لو لم يُختر شيء
        _mark_presented_as_asked(sess, ask_list)
        return enhance_and_localize_response(
            {"sid": sid, "ask": ask_list},
            sess.get("lang", "ar")
        )
    dist = (core_resp.get("result") or {}).get("M1", {}) or {}
    return None, dist


@app.post("/api/chat_m1")
@ensure_json_object

def api_chat_m1():
    sid = get_sid(request, allow_create=True)

    if not sid:
        return {
            "ok": False,
            "error": "sid مفقود",
            "ui_hint": {"require_sid": True}
        }, 400

    sess = SESSIONS[sid]

    data = request.json or {}
    if (
        not data
        and sess.get("stage") == "started"
    ):
        return {
            "sid": sid,
            "ui_hint": {"noop": True}
        }
    original_intent = (data.get("intent") or "")
    msg_raw = data.get("message", "")
    answers_in = data.get("answers") or {}
    
    # ✅ لو رجع جواب لسؤال more_symptoms كـ answers → حوّله إلى intent
    if isinstance(answers_in, dict) and "more_symptoms" in answers_in:
        yn = normalize_yes_no(answers_in.get("more_symptoms"))
        if yn == "yes":
            original_intent = "more_yes"
        elif yn == "no":
            original_intent = "more_no"


    # ===============================
    # (A) حفظ اللغة فقط إذا اختارها المستخدم صراحة
    # ===============================
    if isinstance(answers_in, dict) and "user_language" in answers_in:
        chosen_lang = answers_in["user_language"]
        sess["lang"] = chosen_lang
        sess["asked"].add("user_language")
        sess["stage"] = "started"

        #  FIX: لا تكمّل التدفق
        payload = {"sid": sid, "ask": [build_free_text_question(initial=False, lang=sess.get("lang","ar"))]}
        return enhance_and_localize_response(payload, sess.get("lang", "ar"))
    intent = normalize_intent(original_intent, msg_raw)
    print(f"[CHAT_M1] intent={intent!r} msg={bool(msg_raw)} answers={bool(answers_in)} sid={sid}")

    # ===============================
    # (A.1) استقبال إجابات الشيك بوكس إن وُجدت
    # يدعم الاسمين: symptoms_m1 (الموحد) و symptoms_checkbox (قديم)
    # ===============================

    if isinstance(answers_in, dict) and (
        "symptoms_m1" in answers_in or "symptoms_checkbox" in answers_in
    ):
        # أولاً: وسّع أي حقول أخرى كما المعتاد
        receive_answers_for_sid(sid, answers_in)

        checkbox_selected = answers_in.get("symptoms_m1") or answers_in.get("symptoms_checkbox")
        
        # حالة: المستخدم اختار أعراضًا (قائمة غير فارغة)
        if isinstance(checkbox_selected, list) and checkbox_selected:
            # وحِّد السلوك مع /api/controller عبر دالتك المساعدة
            ask_next, dist = apply_core_checkbox_batch(sid, checkbox_selected, sess)
            if ask_next:
                return ask_next  # (ask_next مُعدّة مسبقًا عبر enhance_and_localize_response)
            # لا توجد أسئلة جديدة → dist يحتوي توزيع M1 إن رجع من core
            # يمكنك هنا متابعة منطق الانتقال إلى M2 بناءً على dist/الكفاية إن رغبت.



        # المستخدم أرسل بدون اختيار أعراض
        if isinstance(checkbox_selected, list) and not checkbox_selected:
            # 1) زد عداد التخطي
            sess["skip_count"] = int(sess.get("skip_count", 0)) + 1

            # 2) اطلب دفعة جديدة من M1 بناءً على asked الحالي


            if sufficient:
                top_dept = max(dist, key=dist.get) if dist else "general"
                disease = DEPT_TO_DISEASE.get(top_dept, top_dept)
                sess["mode"] = "M2"
                sess["target_disease"] = disease
                sess["stage"] = "in_m2"

                q_all = generate_questions("M2", disease_required=[disease], lang=sess.get("lang", "ar"))
                missing = [k for k in get_required_keys_for_disease(disease)
                        if k.lower() not in {kk.lower() for kk in sess["answers"].keys()}]
                q_list = [qq for qq in q_all if (qq.get("name", "").lower() in {mm.lower() for mm in missing})] or q_all
                _mark_presented_as_asked(sess, q_list)

                payload = {
                    "top_department": top_dept,
                    "target_disease": disease,
                    "ask": q_list,
                    "sid": sid,
                    "note": "تم اعتبار الأعراض كافية للانتقال للمرحلة التالية."
                }
                return enhance_and_localize_response(payload, sess.get("lang", "ar"))

            out = generate_questions_from_columns(
                columns=m1_feature_pool(),
                asked_set=_get_asked_set(sess),
                batch_size=5,
                m1_scores=sess.get("M1_scores"),
                answered_map=sess.get("answers", {}),
                add_entire_batch_to_asked=False,
                lang=sess.get("lang", "ar")  # ✅
            )
            if isinstance(out, tuple):
                q_list, asked_updated = out
                _store_asked(sess, set(asked_updated or []))
            else:
                q_list = out

            if q_list:
                ask_out = q_list if isinstance(q_list, list) else [q_list]
                ask_out = _sanitize_ask_list(ask_out)
                _mark_presented_as_asked(sess, ask_out)
                return enhance_and_localize_response({"sid": sid, "ask": ask_out}, sess.get("lang", "ar"))

            # 3) لا توجد دفعات جديدة ولا اكتفاء كافٍ: سلوك واضح غير معلّق
            answers_for_stop = _build_answers_for_stop(sess)
            sufficient, _reason = m1_is_sufficient(answers_for_stop)
            dist = run_M1_on_answers(answers_for_stop) or {}
            if not sufficient:
                if sess["skip_count"] >= 3:
                    payload = {
                        "sid": sid,
                        "ask": [build_free_text_question(initial=False)],
                        "note": "لا يمكن التقدم دون اختيار أي أعراض. اكتب أعراضك نصًا أو اختر من القوائم."
                    }
                    return enhance_and_localize_response(payload, sess.get("lang", "ar"))
                return enhance_and_localize_response(
                    {"sid": sid, "note": "اختر عرضًا واحدًا على الأقل أو اكتب أعراضك نصًا."},
                    sess.get("lang", "ar")
                )





            # =========================
            # PATCH: M1 → M2 (chat_m1)
            # =========================

            if dist:
                answers_for_stop = _build_answers_for_stop(sess)
            

                # ✅ حساب توزيع الأقسام بعد كل checkbox
                current_dist = run_M1_on_answers(answers_for_stop) or {}
                sess["M1_scores"] = current_dist

                # إرسال النتائج الحالية للواجهة مع الأسئلة التالية
                ask_out = q_list if 'q_list' in locals() else []

                # تهيئة payload مؤقت
                payload = {
                    "sid": sid,
                    "ask": ask_out,
                    "M1_distribution": dict(sorted(current_dist.items(), key=lambda x: -x[1]))
                }
                
                # الآن نتحقق من sufficiency للانتقال إلى M2
                sufficient, _ = m1_is_sufficient(answers_for_stop)
                if sufficient:
                    top_dept = max(current_dist, key=current_dist.get)
                    mapped = DEPT_TO_DISEASE.get(top_dept, top_dept)
                    sess["mode"] = "M2"
                    sess["target_disease"] = mapped

                    # توليد كل أسئلة M2 للمرض
                    q_all = generate_questions("M2", disease_required=[mapped], lang=sess.get("lang", "ar"))
                    missing = [
                        k for k in get_required_keys_for_disease(mapped)
                        if k.lower() not in {kk.lower() for kk in sess["answers"].keys()}
                    ]
                    q_list = [qq for qq in q_all if qq.get("name") in missing] or q_all
                    ask_out = q_list if isinstance(q_list, list) else [q_list]
                    _mark_presented_as_asked(sess, ask_out)

                    # تحديث payload النهائي مع الأسئلة الجديدة
                    payload["ask"] = ask_out

                return enhance_and_localize_response(payload, sess.get("lang", "ar"))

                

            # إن لم تكن كافية: نكمل بمنطق M1 أدناه...
        # إذا لم يتحقق أي مما سبق، اسمح للتدفق الحالي بالاستمرار (free text / yes-no ...)

    # ===============================
    # (0) بداية الجلسة: سؤال اختيار اللغة أولاً أو سؤال نص حر
    # ===============================
    if (
        intent == ""
        and not msg_raw
        and not answers_in
        and sess.get("stage", "init") == "init"
        and sess["mode"] == "M1"
    ):
        if not sess.get("lang"):
            payload = {"ask": [build_language_question()], "sid": sid}
            sess["stage"] = "started"
            return payload
        else:
            payload = {"ask": [build_free_text_question(initial=True)], "sid": sid}
            sess["stage"] = "started"
            payload = enhance_and_localize_response(payload, sess.get("lang", "ar"))
            return payload

    # ===============================
    # (1) معالجة النص الحر (أعراض)
    # ===============================
    free_text = None
    if msg_raw:
        free_text = msg_raw
    elif isinstance(answers_in, dict) and "free_text_symptoms" in answers_in:
        free_text = str(answers_in.get("free_text_symptoms") or "").strip()

    if intent in ("", "symptoms") and free_text:
        norm = llm_normalize_user_text(free_text)
        candidate = norm.get("text_ar") or norm.get("text_en") or free_text

        # محاولة استخراج الأعراض من النص
        extracted = {}
        try:
            if NLPAgent and nlp_agent:
                try:
                    extracted = nlp_agent.extract(candidate)
                except Exception as e:
                    print(f"[NLPAgent][EXC] {e}")
        except Exception as e:
            print(f"[NLPAgent][EXC] {e}")
            extracted = {}

        if not extracted:
            extracted = llm_extract_symptoms(candidate, m1_feature_pool()) or {}

        if not extracted:
            payload = {
                "error": "تعذّر تحليل النص تلقائيًا. اذكر أعراضك واحدة واحدة.",
                "ask": [build_free_text_question(initial=False)],
                "sid": sid
            }
            payload = enhance_and_localize_response(payload, sess.get("lang", "ar"))
            return payload

        # تحويل المفاتيح إلى صيغة تدريب M1 ثم تخزين الإجابات
        canon = canonicalize_to_m1_keys(extracted, m1_feature_pool())
        receive_answers_for_sid(sid, canon)
        
        # controller.py (بعد استقبال الأجوبة النصية)
        asked_set = _get_asked_set(sess)
        asked_set.update(canon.keys())
        _store_asked(sess, asked_set)
        sess["asked_symptoms"] = list(asked_set)

        sess["stage"] = "in_m1"
        # تلميح للواجهة ليسأل: هل لديك أعراض أخرى؟
        payload = {"sid": sid, "ask": [build_more_symptoms_yesno(lang=sess.get("lang", "ar"))]}
        payload = enhance_and_localize_response(payload, sess.get("lang", "ar"))
        return payload




    # ===============================
    # (2) نعم / لا — هل لديك أعراض أخرى؟
    # ===============================

    # ---- (2.1) نعم: اطلب نص حر إضافي

    if intent == "more_yes":
        payload = {"ask": [build_free_text_question(initial=False, lang=sess.get("lang", "ar"))], "sid": sid}
        payload = enhance_and_localize_response(payload, sess.get("lang", "ar"))
        return payload

    # ---- (2.2) لا: قيّم الاكتفاء أو اطرح شيك بوكس إضافي (بدون m1_followups)


    if intent == "more_no":
        # 1) أعمدة أعراض M1
        pool = m1_feature_pool() or []

        # 2) نبني "إجابات مُشاهَدة" فقط: أي مفاتيح ضمن الأعمدة أجاب عنها المستخدم (0 أو >0)،
        #    مع تطبيع الأسماء إلى lowercase/strip لتوحيد المقارنات، ثم نرجع للمفتاح الأصلي.
        user_answers = (sess.get("answers") or {})
        # خريطة: اسم مصغّر → الاسم الأصلي في الأعمدة
        pool_lc_map = { (k or "").strip().lower(): k for k in pool }

        observed_answers = {}
        for k, v in user_answers.items():
            kn = (k or "").strip().lower()
            if kn in pool_lc_map and v is not None:
                orig_key = pool_lc_map[kn]
                # 0 = إجابة (لا عرض)، >0 = إيجابي
                observed_answers[orig_key] = v

        # 3) شرط الإيقاف يُطبّق على المُشاهَد فقط:
        #    - 0 يحسب ضمن total_questions لكنه لا يحسب ضمن positive_count
        answers_for_stop = _build_answers_for_stop(sess)
        sufficient, _reason = m1_is_sufficient(answers_for_stop, max_questions=25, min_positive=3)

        # 4) توزيع الأقسام يُحسب على المُشاهَد فقط (أدق من الكثيف)
# 4) حساب توزيع الأقسام باستخدام كل الإجابات (تضم الصفر للأعراض التي لم تُختَر)
        answers_for_stop = _build_answers_for_stop(sess)
        dist = run_M1_on_answers(answers_for_stop) or {}
        sess["M1_scores"] = dist  # تخزين في الجلسة

        # 5) إذا تحقق الاكتفاء → انتقل إلى M2
        if sufficient:
            # اختيار القسم الأعلى إن توفر dist، وإلا احتياط 'general'
            if dist:
                top_dept = max(dist, key=dist.get)
            else:
                top_dept = "general"

            disease = DEPT_TO_DISEASE.get(top_dept, top_dept)

            sess["mode"] = "M2"
            sess["target_disease"] = disease
            sess["stage"] = "in_m2"

            q_all = generate_questions("M2", disease_required=[disease], lang=sess.get("lang", "ar"))

            # اطرح فقط الأسئلة الناقصة لهذا المرض
            answered_keys_lc = { (k or "").lower() for k in (sess.get("answers") or {}).keys() }
            required_lc = { (k or "").lower() for k in get_required_keys_for_disease(disease) }
            missing_lc  = required_lc - answered_keys_lc

            q_list = [qq for qq in q_all if (qq.get("name", "").lower() in missing_lc)] or q_all

            payload = {
                "top_department": top_dept,
                "target_disease": disease,
                "ask": q_list,
                "sid": sid,
                "note": "تم اعتبار الأعراض كافية للانتقال للمرحلة التالية."
            }
            return enhance_and_localize_response(payload, sess.get("lang", "ar"))

        # 6) خلاف ذلك: تابع بدفعات M1 من المصدر الوحيد (مع منع التكرار التام)
        #    حراسة نوع asked_set (قائمة ← set) قبل التمرير
        asked_set = _get_asked_set(sess)



        out = generate_questions_from_columns(
            columns=pool,
            asked_set=asked_set,
            batch_size=5,
            m1_scores=dist,
            answered_map=sess.get("answers", {}),  # ✅
            add_entire_batch_to_asked=False,       # ✅
            lang=sess.get("lang", "ar")  # ✅
        )

        # فكّ الإخراج بأمان
        if isinstance(out, tuple):
            question, asked = out
        else:
            question, asked = out, asked_set

        # 7) لا مزيد من الأعراض للسؤال: انتقل إلى M2 (سياسة احتياط) + تنبيه
        if question is None:
            top_dept = max(dist, key=dist.get) if dist else "general"
            disease  = DEPT_TO_DISEASE.get(top_dept, top_dept)

            sess["mode"] = "M2"
            sess["target_disease"] = disease
            sess["stage"] = "in_m2"

            q_all = generate_questions("M2", disease_required=[disease], lang=sess.get("lang", "ar"))

            payload = {
                "top_department": top_dept,
                "target_disease": disease,
                "ask": q_all,
                "sid": sid,
                "note": "لا توجد أعراض إضافية للسؤال. انتقلنا للمرحلة التالية، وقد تتأثر الدقة لنقص المعلومات."
            }
            return enhance_and_localize_response(payload, sess.get("lang", "ar"))

        # 8) ما زالت هناك دفعة أسئلة M1 — أرسلها للمستخدم
        # خزّن asked كقائمة (JSON-friendly) مع إزالة التكرار
        _store_asked(sess, set(asked or []))

        payload = {
            "ask": [question],   # الدالة تُعيد سؤالًا واحدًا (checkbox) يحوي عدة خيارات
            "sid": sid,
            "note": "نحتاج لبعض المعلومات الإضافية قبل التقييم التالي."
        }
        return enhance_and_localize_response(payload, sess.get("lang", "ar"))

    # ===============================
    # الافتراضي: أعِد سؤال نص حر أولي/إضافي حسب اللغة
    # ===============================

    # حماية إضافية إن رغبت الإبقاء على الذيل
    if 'payload' not in locals():
        payload = {"sid": sid, "ui_hint": {"noop": True}}

    payload = enhance_and_localize_response(payload, sess.get("lang", "ar"))
    return payload
# ==============================
# إعادة ضبط جلسة sid
# ==============================
@app.post("/api/reset_sid")
@ensure_json_object
def api_reset_sid():
    sid = request.headers.get("X-Session-Id") or request.args.get("sid") or request.cookies.get("sid")
    if sid and sid in SESSIONS:
        SESSIONS[sid] = {
            "answers": {},
            "pending_questions": [],
            "mode": "M1",
            "target_disease": None,
            "asked": (),
            "lang": None,
            "last_user_text": "",
            "stage": "init",
            "skip_count": 0


        }
        return {"ok": True, "sid": sid}
    return {"ok": False, "error": "sid غير موجود"}

# ==============================
# معالج أخطاء عام — يعيد JSON بدل HTML
# ==============================
@app.errorhandler(Exception)
def handle_ex(e):
    print(f"[SERVER][EXC] {e}")
    return jsonify({"ok": False, "error": str(e)}), 500

# ==============================
# تشغيل Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)