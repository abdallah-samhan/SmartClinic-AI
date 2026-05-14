# -*- coding: utf-8 -*-
"""
اختبار مُعاد كتابة يناسب نسخة core.py الحالية (بدون core_step)
- يبدأ بـ core_start_conversation
- يكرّر النداء إلى core_handle_answers كما لو أن مريضًا يجيب على أسئلة M1
- يتوقف عند عدم وجود أسئلة جديدة ويطبع النتيجة النهائية (درجات M1)
طريقة التشغيل:
    python test_core_conversation_v4.py
"""

import json
import sys

try:
    import core
except ImportError:
    print("❌ لم يتم العثور على core.py في المسار الحالي.")
    sys.exit(1)


def show(title, data):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    try:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(data)


def pick_values_from_options(options, already=None, max_pick=3):
    """اختيار حتى 3 قيم من الخيارات مع إزالة التكرارات وتجنب العناصر المختارة سابقًا."""
    already = already or set()
    chosen = []
    seen = set()
    for opt in options or []:
        # خيار قد يكون dict بقيمة 'value' أو نصًا مباشرة
        if isinstance(opt, dict):
            val = opt.get('value', opt.get('label'))
        else:
            val = opt
        if val is None:
            continue
        if val in seen or val in already:
            continue
        seen.add(val)
        chosen.append(val)
        if len(chosen) >= max_pick:
            break
    return chosen


def main():
    # 1) بدء المحادثة
    start = core.core_start_conversation({"sid": "TEST-CORE-V4-001"})
    show("🔹 بدء المحادثة", start)

    sid = start.get("sid")
    session = start.get("session", {})

    # سنضيف إجابات أولية لمحاكاة رسالة المريض الأولى (نص حر) عبر قائمة افتراضية
    # بالرغم أن السؤال الأول من نوع نص، سنمرّر أعراضًا كبداية لدفع التدفّق.
    initial_symptoms = ["Cough", "fatigue_general", "difficulty_breathing"]
    resp = core.core_handle_answers({
        "sid": sid,
        "session": session,
        "answers": {"symptoms_checkbox": initial_symptoms}
    })
    show("🔹 بعد الإجابات الأولية", resp)

    session = resp.get("session", {})

    # 2) حلقة تكرارية على دفعات أسئلة M1
    max_rounds = 15
    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1
        ask_list = resp.get("ask")
        if not ask_list:
            break

        # في هذه النسخة، ask_list عبارة عن قائمة فيها سؤال واحد (checkbox)
        q = ask_list[0]
        options = q.get("options", [])
        already = set(session.get("asked_symptoms", []))
        picked = pick_values_from_options(options, already=already, max_pick=3)

        show(f"🟦 الجولة {round_idx}: السؤال المطروح", q)
        show(f"🟦 الجولة {round_idx}: القيم المختارة", picked)

        # مرّر الإجابات إلى core_handle_answers
        payload = {
            "sid": sid,
            "session": session,
            "answers": {"symptoms_checkbox": picked}
        }
        resp = core.core_handle_answers(payload)
        show(f"🟦 الجولة {round_idx}: ناتج بعد الإجابات", resp)

        session = resp.get("session", {})

    # 3) النتيجة النهائية
    final = resp.get("result") or {}
    if final:
        show("🏁 النتيجة النهائية (درجات M1)", final)
    else:
        show("⚠️ لم تُرجع النتيجة النهائية — قد تكون ميّزات M1/الملفات غير متوفّرة", resp)

    print("\n=== انتهى الاختبار ===")


if __name__ == '__main__':
    main()
