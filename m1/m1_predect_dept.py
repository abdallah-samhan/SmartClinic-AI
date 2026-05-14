
# m1_predect_dept.py — نسخة مُصحّحة ومُحسّنة
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd

class M1DeptPredictor:
    def __init__(self, pkl_path: str = "m1/dept_models.pkl", min_required: int = 4, effective_top_n: int = 50):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"لم يتم العثور على الحزمة: {pkl_path}")
        with open(pkl_path, "rb") as f:
            PKL = pickle.load(f)

        # تحميل نماذج الحزمة
        self.rf_cal = PKL.get("rf_cal")
        self.gb_cal = PKL.get("gb_cal")
        self.et_cal = PKL.get("et_cal")
        self.stack = PKL.get("stack")
        self.extra = PKL.get("extra_estimators", [])
        self.le = PKL.get("label_encoder")
        self.best_w = PKL.get("best_weights", {})
        self.feature_cols = PKL.get("feature_cols", [])

        # حواجز أمان لـ label_encoder
        if self.le is None or getattr(self.le, "classes_", None) is None or len(getattr(self.le, "classes_", [])) == 0:
            raise ValueError("LabelEncoder/classes_ مفقودة أو فارغة في PKL.")
        self.n_classes = len(self.le.classes_)

        self.min_required = int(min_required)

        # تجميع النماذج للمزج الوزني
        self.models_for_blend = {"rf": self.rf_cal, "gb": self.gb_cal, "et": self.et_cal}
        for name, mdl in self.extra:
            self.models_for_blend[name] = mdl

        # مجموعات الكلمات للمؤشرات
        self.GROUPS = {
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

        # أعمدة مشتقة + الأساسية
        self.derived_cols = ["count_positive", "pct_positive"] + list(self.GROUPS.keys())
        original_base = [c for c in self.feature_cols if c not in self.derived_cols]

        # === تفضيل "خصائص فعّالة" إن وُجدت ===
        effective = PKL.get("effective_feature_cols") or PKL.get("selected_features") or []
        base_dir = os.path.dirname(os.path.abspath(pkl_path))
        txt_path = os.path.join(base_dir, "effective_m1_keys.txt")
        if not effective and os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    effective = [ln.strip() for ln in f if ln.strip()]
            except Exception:
                effective = []

        if effective:
            self.original_feature_cols = [c for c in effective if c in original_base]
        else:
            self.original_feature_cols = original_base

        # خريطة Lower -> Original لتقبّل إجابات بحروف صغيرة من الطبقات العليا
        self.original_feature_cols_lower_map = {c.lower(): c for c in self.original_feature_cols}

    # واجهات الكنترولر
    def needs_more_questions(self, answers_dict) -> bool:
        return len(answers_dict or {}) < self.min_required

    def add_group_index(self, X: pd.DataFrame, features: list, name: str, keywords: list):
        cols = np.array(features)
        mask = np.zeros(len(cols), dtype=bool)
        for kw in keywords:
            mask |= np.array([kw.lower() in c.lower() for c in cols])
        X[name] = X.loc[:, cols[mask]].sum(axis=1)
        return X

    def build_feature_vector(self, answers_dict: dict) -> pd.DataFrame:
        row = {c: 0 for c in self.feature_cols}
        for k, v in (answers_dict or {}).items():
            # قبول مفاتيح lower عبر الخريطة إلى الاسم الأصلي
            orig = self.original_feature_cols_lower_map.get(str(k).lower())
            if orig:
                try:
                    row[orig] = float(v)
                except Exception:
                    row[orig] = 0
        X = pd.DataFrame([row])

        # حساب المشتقات على الأعمدة الأصلية فقط
        X["count_positive"] = X.loc[:, self.original_feature_cols].sum(axis=1)
        X["pct_positive"]   = X["count_positive"] / (len(self.original_feature_cols) if self.original_feature_cols else 1)
        for gname, kws in self.GROUPS.items():
            X = self.add_group_index(X, self.original_feature_cols, gname, kws)
        return X

    def proba_in_le_order(self, model, X: pd.DataFrame):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim != 2:
            raise ValueError("predict_proba يجب أن تُرجع مصفوفة ثنائية الأبعاد (batch, n_classes_model).")

        est_classes = getattr(model, "classes_", None)
        aligned = np.zeros((proba.shape[0], self.n_classes), dtype=float)
        if est_classes is None:
            aligned[:, :proba.shape[1]] = proba
            return aligned

        est_classes = np.array(est_classes)
        if est_classes.dtype.kind in {"U", "S", "O"}:
            est_idx = self.le.transform(est_classes)
        else:
            est_idx = est_classes.astype(int)

        aligned[:, est_idx] = proba
        return aligned

    def blend_proba(self, X: pd.DataFrame, weights: dict):
        """تصحيح المزج: يبدأ بـ None، ويتحوّل إلى المصفوفة الأولى، مع fallback إذا لم يتم المزج."""
        proba_sum = None
        total_w = 0.0
        for key, mdl in self.models_for_blend.items():
            w = float(weights.get(key, 0.0))
            if mdl is None or w <= 0:
                continue
            proba = self.proba_in_le_order(mdl, X)  # [batch, n_classes]
            proba_sum = proba if proba_sum is None else (proba_sum + w * proba)
            total_w += w

        if proba_sum is None:
            # لا مزج فعليًا: استخدم المكدّس كـ fallback
            return self.proba_in_le_order(self.stack, X)

        return proba_sum / max(total_w, 1e-9)

    def predict_top(self, answers_dict, top: int = 5):
        X = self.build_feature_vector(answers_dict)
        if self.best_w and isinstance(self.best_w, dict):
            proba = self.blend_proba(X, self.best_w)
        else:
            proba = self.proba_in_le_order(self.stack, X)

        proba = np.asarray(proba).reshape(-1, self.n_classes)[0]
        idx_sorted = np.argsort(proba)[::-1]
        labels = list(self.le.classes_)
        return [
            {"department": labels[i], "prob": float(proba[i])}
            for i in idx_sorted[:max(1, top)]
        ]

    def predict_dict(self, answers_dict, top: int = 5):
        return {r["department"]: r["prob"] for r in self.predict_top(answers_dict, top=top)}
