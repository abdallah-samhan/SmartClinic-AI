# test_m1_predictor.py
# -*- coding: utf-8 -*-

from m1_predect_dept import M1DeptPredictor
import random

# تحميل النموذج
predictor = M1DeptPredictor(pkl_path="m1/dept_models.pkl")

# تعريف مجموعة من الأعراض المحتملة (أعمدة النموذج الأصلية)
sample_features = predictor.original_feature_cols

# دالة لتوليد إجابات عشوائية
def generate_random_answers(features, num_positive=None):
    if num_positive is None:
        num_positive = random.randint(0, len(features))
    positives = random.sample(features, num_positive)
    answers = {f: 1 if f in positives else 0 for f in features}
    return answers

# عدد الاختبارات التي نريد تشغيلها
num_tests = 10

for i in range(1, num_tests + 1):
    answers = generate_random_answers(sample_features)
    top_preds = predictor.predict_top(answers, top=5)
    
    print(f"\n=== Test {i} ===")
    print("Answers (sample positive keys):", [k for k, v in answers.items() if v])
    for rank, pred in enumerate(top_preds, 1):
        print(f"{rank}. Dept: {pred['department']}, Probability: {pred['prob']:.3f}")
