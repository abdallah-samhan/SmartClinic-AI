
# -*- coding: utf-8 -*-
# llm_utils.py — أدوات OpenRouter + اللغة (تطبيع/ترجمة/تعاطف/NLG/استخراج احتياطي)
import os, json, re, requests
from functools import lru_cache

# ==============================
# إعدادات OpenRouter — استخدم متغيّر بيئي بدل مفتاح مكشوف
# ==============================
# Load API key from environment variable for security
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL   = "mistralai/mistral-7b-instruct:free"

# ==============================
# أعلام الميزات
# ==============================
USE_LLM_FALLBACK = True
USE_LLM_TONE = True
USE_LLM_NORM = True
USE_LLM_TRANSLATE= True
USE_LLM_NLG_Q = True
USE_LLM_NLG_RESULT = True
DEFAULT_UI_LANG = "ar"

# مفاتيح احتياطية إن كانت مفاتيح M1 فارغة
DEFAULT_FALLBACK_M1_KEYS = [
    "chp","breathing_problem","htn","ihd","fatigue","headache","cough","dry_cough","wheezing",
    "diabetes","blood_glucose_level","bmi","hypertension","heart_disease","age","gender","sex",
    "fever","sore_throat","running_nose","chronic_lung_disease","asthma","hyper_tension",
    "smoke","active","lifestyle","hr","height","weight","bpsys","bpdias","years","ecgpatt"
]

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670-\u06ED]")

def detect_lang_simple(txt: str) -> str:
    if re.search(r"[\u0600-\u06FF]", txt or ""):
        return "ar"
    if re.search(r"[A-Za-z]", txt or ""):
        return "en"
    return "ar"

def _llm_call(messages, temperature=0.0, extra=None):
    if not OPENROUTER_API_KEY:
        return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": float(temperature),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=25)
        if resp.status_code != 200:
            print(f"[LLM][HTTP-{resp.status_code}] {resp.text[:500]}")
            return None
        data = resp.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[LLM][EXC] {e}")
        return None

def _strip_diacritics(txt: str) -> str:
    return _AR_DIACRITICS.sub("", txt or "")

def _normalize_ar_letters(txt: str) -> str:
    t = (txt or "")
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ؤ","و").replace("ئ","ي")
    t = t.replace("ة","ه").replace("ى","ي")
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return t

_COMMON_AR_MISSPELL = [
    (r"\bالحراره\b", "حرارة"),
    (r"\bالغثايان\b", "غثيان"),
]
def _rule_fix_common_ar_misspell(txt: str) -> str:
    t = txt or ""
    for pat, repl in _COMMON_AR_MISSPELL:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t

def llm_normalize_user_text(message: str) -> dict:
    if OPENROUTER_API_KEY and (message or "").strip():
        system = "You normalize noisy user text (Arabic/English) for medical symptom extraction. Respond in strict JSON."
        user = f"""
أعد صياغة/تصحيح النص التالي بوضوح دون تغيير المعنى. قدم نسختين: عربية وإنجليزية إن أمكن.
أعد فقط JSON:
{{ 
  "lang": "ar" أو "en",
  "text_ar": "نسخة عربية مناسبة للاستخراج",
  "text_en": "نسخة إنجليزية مناسبة للاستخراج"
}}
النص:
\"\"\"{(message or '').strip()}\"\"\"""".strip()
        out = _llm_call([
            {"role":"system","content":system},
            {"role":"user","content":user}
        ], temperature=0.0)
        try:
            if not out: raise ValueError("No LLM output")
            m = re.search(r"\{.*\}", out, re.S)
            obj = json.loads(m.group(0)) if m else json.loads(out)
            lang = obj.get("lang") or detect_lang_simple(message)
            return {
                "lang": "ar" if str(lang).lower().startswith("ar") else "en",
                "text_ar": obj.get("text_ar","") or "",
                "text_en": obj.get("text_en","") or ""
            }
        except Exception as e:
            print(f"[NORM-LLM][PARSE] {e}\n RAW: {out[:200] if out else 'None'}")
            lang = detect_lang_simple(message)
            if lang == "ar":
                cleaned = _rule_fix_common_ar_misspell(_normalize_ar_letters(_strip_diacritics(message)))
                return {"lang": "ar", "text_ar": cleaned, "text_en": ""}
            else:
                return {"lang": "en", "text_en": (message or "").strip(), "text_ar": ""}

    # بدون LLM
    lang = detect_lang_simple(message)
    if lang == "ar":
        cleaned = _rule_fix_common_ar_misspell(_normalize_ar_letters(_strip_diacritics(message)))
        return {"lang": "ar", "text_ar": cleaned, "text_en": ""}
    else:
        return {"lang": "en", "text_en": (message or "").strip(), "text_ar": ""}

def llm_translate(text: str, target_lang: str) -> str:
    if not (OPENROUTER_API_KEY and (text or "").strip()):
        return text
    if target_lang not in {"ar","en"}:
        return text
    system = "You are a concise medical chatbot translator."
    user = f"ترجم بتصرف طبي بسيط إلى لغة { 'العربية' if target_lang=='ar' else 'English' } فقط دون شروح:\n{text}"
    out = _llm_call([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ], temperature=0.0)
    return out.strip() if out else text

def llm_empathize_simplify(text: str, target_lang: str = "ar") -> str:
    if not (OPENROUTER_API_KEY and (text or "").strip()):
        return text
    system = "Rewrite with a friendly, empathetic, medically-neutral tone. Keep it short."
    tgt_name = "العربية" if target_lang == "ar" else "English"
    user = f"أعد صياغة الرسالة التالية بنبرة متعاطفة ومباشرة بلغة {tgt_name} فقط:\n{text}"
    out = _llm_call([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ], temperature=0.3)
    return out.strip() if out else text

def llm_extract_symptoms(message: str, m1_keys: list) -> dict:
    if not (OPENROUTER_API_KEY and (message or "").strip()):
        return {}
    allowed = sorted(list(set(m1_keys or [])))
    if not allowed:
        allowed = DEFAULT_FALLBACK_M1_KEYS[:]
    allowed_preview = ", ".join(allowed)

    system_msg = (
        "You are an information extraction component. "
        "Extract ONLY the presence (1) or absence (0) of each symptom. Output STRICT JSON."
    )
    user_instruction = f"""
حوّل النص التالي إلى JSON يحتوي فقط على مفاتيح من لائحة مفاتيح التدريب (M1 keys) وقيمها 0 أو 1.
- إذا ذُكر العرض بصيغة إيجابية = 1، وإذا نُفي أو غير مذكور = 0.
- لا تُخرج أي نص خارج JSON.
- المفاتيح المسموحة (M1 keys): {allowed_preview}
النص:
\"\"\"{(message or '').strip()}\"\"\" 
أعد فقط JSON بهذا الشكل:
{{ 
  "symptoms": {{
    "<m1_key>": 0 أو 1
  }}
}}
""".strip()

    out = _llm_call([
        {"role":"system","content":system_msg},
        {"role":"user","content":user_instruction}
    ], temperature=0.0)
    try:
        if not out: return {}
        m = re.search(r"\{.*\}", out, re.S)
        obj = json.loads(m.group(0)) if m else json.loads(out)
        sym = obj.get("symptoms", obj)
        out_d = {}
        m1set = set(k.lower() for k in (m1_keys or allowed))
        for k, v in (sym or {}).items():
            kl = k.lower().strip()
            if kl in m1set:
                if isinstance(v, (int, float)):
                    out_d[kl] = 1 if float(v) >= 0.5 else 0
                else:
                    s = str(v).strip().lower()
                    out_d[kl] = 1 if s in {"1","true","yes","y","نعم"} else 0
        return out_d
    except Exception as e:
        print(f"[LLM-FALLBACK][PARSE] {e}\n RAW: {out[:200] if out else 'None'}")
        return {}

@lru_cache(maxsize=256)
def _rewrite_questions_cached(key: str) -> list:
    try:
        payload = json.loads(key)
    except Exception:
        return []
    target_lang = payload.get("lang","ar")
    items = payload.get("items", [])
    simple = [{"q": it.get("q",""), "type": it.get("type",""), "name": it.get("name","")} for it in items]

    system = (
        "You rewrite question prompts for a medical triage UI. "
        "Keep order; preserve meaning; be concise & empathetic; output STRICT JSON array of strings."
    )
    lang_label = "العربية" if target_lang == "ar" else "English"
    user = f"""
أعد صياغة عناصر قائمة الأسئلة التالية بلغة {lang_label} فقط، بنبرة ودودة وواضحة:
- لا تغيّر المعنى أو نوع السؤال.
- اختصر عند الإمكان.
- أعد فقط JSON (مصفوفة نصوص q) بنفس الترتيب.
الأسئلة:
{json.dumps(simple, ensure_ascii=False, indent=2)}
""".strip()
    out = _llm_call([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ], temperature=0.2)
    try:
        if not out: return []
        m = re.search(r"\[.*\]", out, re.S)
        arr = json.loads(m.group(0)) if m else json.loads(out)
        if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
            return []
        return arr
    except Exception as e:
        print(f"[NLG-Q][PARSE] {e}\n RAW: {out[:200] if out else 'None'}")
        return []

def nlg_rewrite_questions(ask_list: list, target_lang: str = "ar") -> list:
    if not (OPENROUTER_API_KEY and isinstance(ask_list, list) and ask_list):
        return ask_list
    items = [{"name": q.get("name",""), "type": q.get("type",""), "q": q.get("q","")}
             for q in ask_list if isinstance(q, dict)]
    cache_key = json.dumps({"lang": target_lang, "items": items}, ensure_ascii=False)
    rewritten = _rewrite_questions_cached(cache_key)
    if rewritten and len(rewritten) == len(items):
        out = []
        idx = 0
        for q in ask_list:
            if not isinstance(q, dict):
                out.append(q); continue
            qq = dict(q)
            qq["q"] = rewritten[idx].strip()
            idx += 1
            out.append(qq)
        return out
    return ask_list

def nlg_format_result_summary(disease: str, percentage: float, target_lang: str = "ar") -> dict:
    level = "low"
    if percentage >= 60: level = "high"
    elif percentage >= 30: level = "medium"

    if not OPENROUTER_API_KEY:
        title = "النتيجة المبدئية" if target_lang=="ar" else "Preliminary Result"
        summary = f"احتمال «{disease}»: {percentage:.1f}%."
        next_steps = "هذه نتيجة مساعدة وليست تشخيصًا. إذا كانت الأعراض مزعجة أو تزداد، يُفضّل مراجعة مختص."
        disclaimer = "لا تستخدم هذه النتيجة لاتخاذ قرارات طبية طارئة."
        return {"title": title, "summary": summary, "next_steps": next_steps, "disclaimer": disclaimer, "level": level}

    system = (
        "You are a medical UX writer. Produce concise, empathetic triage summaries. "
        "Output STRICT JSON: title, summary, next_steps, disclaimer."
    )
    lang_label = "العربية" if target_lang == "ar" else "English"
    user = f"""
اكتب نتيجة موجزة بلغة {lang_label} فقط:
- المرض: {disease}
- النسبة: {percentage:.1f}%
- المستوى: {level}
قواعد:
- لا تغيّر الأرقام أو المعنى.
- لا تصف علاجًا مباشرًا؛ فقط خطوة قادمة عامة ومتى يلزم طلب رعاية.
- أعد JSON فقط بهذه الحقول.
""".strip()
    out = _llm_call([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ], temperature=0.2)
    try:
        if not out: raise ValueError("No LLM output")
        m = re.search(r"\{.*\}", out, re.S)
        obj = json.loads(m.group(0)) if m else json.loads(out)
        for k in ("title","summary","next_steps","disclaimer"):
            if k not in obj or not isinstance(obj[k], str):
                raise ValueError("Missing fields")
        obj["level"] = level
        return obj
    except Exception as e:
        print(f"[NLG-RES][PARSE] {e}\n RAW: {out[:200] if out else 'None'}")
        title = "النتيجة المبدئية" if target_lang=="ar" else "Preliminary Result"
        summary = f"احتمال «{disease}»: {percentage:.1f}%."
        next_steps = "هذه نتيجة مساعدة وليست تشخيصًا."
        disclaimer = "لا تستخدم هذه النتيجة لاتخاذ قرارات طبية طارئة."
        return {"title": title, "summary": summary, "next_steps": next_steps, "disclaimer": disclaimer, "level": level}



def enhance_and_localize_response(payload: dict, target_lang: str):
    """
    تحسين محلي للرد:
    - يعيد كتابة نصوص الأسئلة بقالب متعاطف (مرة واحدة فقط).
    - يبسط ويترجم نصوص ask_free_text و note.
    - ✅ يجب أن يُعيد دائمًا payload (حتى لا يصبح None).
    """
    if not payload:
        return payload

    # إعادة كتابة الأسئلة (مرة واحدة)
    if "ask" in payload and isinstance(payload["ask"], list) and payload["ask"]:
        try:
            payload["ask"] = nlg_rewrite_questions(payload["ask"], target_lang=target_lang)
        except Exception as e:
            print(f"[NLG-Q][ERR] {e}")

    # تبسيط/تعاطف + ترجمة لنص حر اختياري
    if "ask_free_text" in payload and isinstance(payload["ask_free_text"], str):
        txt = payload["ask_free_text"]
        txt = llm_empathize_simplify(txt, target_lang=target_lang)
        txt = llm_translate(txt, target_lang) if detect_lang_simple(txt) != target_lang else txt
        payload["ask_free_text"] = txt

    # تبسيط/تعاطف + ترجمة للملاحظات
    if "note" in payload and isinstance(payload["note"], str):
        txt = payload["note"]
        txt = llm_empathize_simplify(txt, target_lang=target_lang)
        txt = llm_translate(txt, target_lang) if detect_lang_simple(txt) != target_lang else txt
        payload["note"] = txt

    # لا نعيد صياغة q لكل عنصر مرة ثانية لتقليل النداءات المكررة
    # ✅ مهم: أعد الكائن دائمًا (كان مفقودًا سابقًا)
    return payload
