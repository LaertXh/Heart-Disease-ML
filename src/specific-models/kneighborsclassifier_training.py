from imports import *
from data_cleaning import X_train, y_train, X_test, y_test

model = KNeighborsClassifier(n_neighbors=16, algorithm='brute', metric='cosine',
                             weights='distance')
model.fit(X_train, y_train)
# Calculate and display model accuracy
y_pred = model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)
print(f"Accuracy = {round((accuracy_knn * 100), 3)}%")