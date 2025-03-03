from imports import *
from data_cleaning import X_train, y_train, X_test, y_test

rf = RandomForestClassifier(n_estimators=150, max_features= 'log2', criterion='entropy', random_state=42)
rf.fit(X_train, y_train)
# Calculate and display model accuracy
y_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy = {round((accuracy_rf * 100), 3)}%")