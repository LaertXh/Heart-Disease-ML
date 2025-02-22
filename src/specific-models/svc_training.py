from imports import *
from data_cleaning import X_train, y_train, X_test, y_test

svc = SVC(C=0.001, class_weight='balanced',  kernel='poly', probability=True,
        break_ties=True, random_state=42)

svc.fit(X_train, y_train)

# Calculate and display model accuracy
y_pred = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy = {round((accuracy_svc * 100), 3)}%")