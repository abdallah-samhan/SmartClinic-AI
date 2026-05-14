
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# 1. تح
# ميل البيانات
try:
    data_set = pd.read_csv('C:\xampp\htdocs\web_doctor-main\web_doctor-heart-disease\heart_disease2\heart.csv')
    print("تم تحميل البيانات بنجاح!")
except Exception as e:
    print(f"خطأ في تحميل البيانات: {e}")
    exit()

    # 2. تحديد الأعمدة المطلوبة
cols = ['age', 'sex', 'smoke', 'years', 'chp', 'height', 'weight', 'fh', 'active', 
        'lifestyle', 'ihd', 'hr', 'dm', 'bpsys', 'bpdias', 'htn', 'ecgpatt']

# 3. تحضير البيانات
X = data_set[cols].values
y = data_set.iloc[:, -1].values  # العمود الأخير هو التصنيف



#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)#random state !=0?

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42
)
#5??????????????????????????????????????????????????????????
# 5. تطبيع البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. تدريب النموذج
clf = RandomForestClassifier(
    n_estimators=100,
   max_depth=8,
   random_state=42
)
clf.fit(X_train, y_train)

#clf = DecisionTreeClassifier(
  #  max_depth=5,          # للتحكم في تعقيد النموذج
   # min_samples_split=10, # الحد الأدنى للعينات في العقدة قبل التقسيم
    #class_weight='balanced', # لمعالجة اختلال التوازن
    #random_state=42
#)
#clf.fit(X_train, y_train)

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
# 9. اختبار النموذج على بيانات جديدة
test_samples = [
    [55, 1, 1, 15, 1, 170, 85, 1, 0, 1, 1, 90, 1, 160, 100, 1, 1],  # مريض
    [40, 0, 0, 0, 4, 175, 70, 0, 1, 3, 0, 75, 0, 120, 80, 0, 4]      # سليم
]
print("نتائج الاختبار:")
for sample in test_samples:
    sample_scaled = scaler.transform([sample])
    prediction = clf.predict(sample_scaled)[0]
    print(f"العينة: {sample[:3]}... → النتيجة: {'مريض' if prediction == 1 else 'سليم'}")