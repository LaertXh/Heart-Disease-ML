from imports import *

model = LogisticRegression(penalty='l1', solver='liblinear', class_weight={0: 2, 1: 1}, random_state=42)
model.fit(X_train, y_train)
# Calculate and display model accuracy
predictions = model.predict(X_test)
accuracy_log = model.score(X_test, y_test)
print("accuracy = ", round((accuracy_log * 100), 3), "%")