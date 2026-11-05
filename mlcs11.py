
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create sample database records
data = {
    "record_length": [
        20, 25, 30, 22, 28, 35, 24, 27, 32, 21,
        100, 120, 150, 110, 130, 140, 160, 125, 115, 145
    ],
    "special_characters": [
        0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
        8, 10, 12, 9, 11, 13, 15, 10, 9, 14
    ],
    "sql_keywords": [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        4, 5, 6, 4, 5, 7, 6, 5, 4, 7
    ],
    "failed_attempts": [
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        5, 6, 8, 5, 7, 9, 8, 6, 7, 9
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Separate input features and output labels
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create and train Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Predict test records
y_pred = model.predict(X_test)

# Display predictions
print("Decision Tree Classification Results")
print("------------------------------------")

for actual, predicted in zip(y_test, y_pred):
    if predicted == 0:
        result = "Normal"
    else:
        result = "Malicious"

    print("Actual:", "Normal" if actual == 0 else "Malicious",
          "| Predicted:", result)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Malicious"],
    zero_division=0
))

# Test a new database record
new_record = pd.DataFrame([{
    "record_length": 140,
    "special_characters": 12,
    "sql_keywords": 6,
    "failed_attempts": 8
}])

prediction = model.predict(new_record)

print("New Record Prediction:")
if prediction[0] == 0:
    print("Normal Database Record")
else:
    print("Suspicious / Malicious Database Record")
