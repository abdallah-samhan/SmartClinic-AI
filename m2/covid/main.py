import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. تحميل البيانات
try:
    # استخدم raw string أو مسار أمامي لتجنب مشاكل الـ escape
    data_set = pd.read_csv('C:\\xampp\htdocs\\web_doctor-main\\web_doctor-covid\\covid\\Covid.csv')
    print("تم تحميل البيانات بنجاح!")
    print(f"عدد الصفوف: {len(data_set)}")
except Exception as e:
    print(f"خطأ في تحميل البيانات: {e}")
    exit()

data_set.columns = data_set.columns.str.strip()

# 2. تحويل البيانات النصية إلى رقمية (إذا لزم الأمر)
# تحويل Yes/No إلى 1/0
data_set = data_set.replace({'Yes': 1, 'No': 0})

# 3. تحضير البيانات
X = data_set.iloc[:, :-1].values  # كل الأعمدة ما عدا الأخير
y = data_set.iloc[:, -1].values   # العمود الأخير هو التصنيف

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42,  # 42 شائعة الاستخدام ولكن يمكن تغييرها لأي قيمة
    stratify=y       # للحفاظ على توزيع الفئات متوازن
)

# 5. تطبيع البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. تدريب النموذج
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight='balanced'  # لمعالجة اختلال التوازن إن وجد
)
clf.fit(X_train, y_train)

# 7. تقييم النموذج
y_pred = clf.predict(X_test)

print("\n" + "="*50)
print("تقرير التصنيف:")
print(classification_report(y_test, y_pred))
print("مصفوفة الارتباك:")
print(confusion_matrix(y_test, y_pred))
print(f"الدقة الكلية: {accuracy_score(y_test, y_pred):.2f}")
print("="*50 + "\n")

# 8. حفظ النموذج والمعالج
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("تم حفظ النموذج والمعالج بنجاح!")

# 9. مثال لاختبار النموذج على بيانات جديدة
# تحضير بيانات جديدة بنفس شكل بيانات التدريب
new_data = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0]])  # مثال لبيانات مريض
new_data_scaled = scaler.transform(new_data)
prediction = clf.predict(new_data_scaled)
print(f"نتيجة التنبؤ للبيانات الجديدة: {'إصابة' if prediction[0] == 1 else 'غير مصاب'}")
print(data_set.columns)
