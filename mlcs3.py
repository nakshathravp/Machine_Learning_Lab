
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create sample network dataset

data = {
    "duration": [10, 12, 15, 20, 25, 100, 110, 120, 130, 140,
                 200, 210, 220, 230, 240, 300, 310, 320, 330, 340],

    "packets": [20, 25, 30, 35, 40, 500, 550, 600, 650, 700,
                800, 850, 900, 950, 1000, 50, 55, 60, 65, 70],

    "failed_logins": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 10, 12, 15, 18, 20],

    "label": [
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "DoS", "DoS", "DoS", "DoS", "DoS",
        "DDoS", "DDoS", "DDoS", "DDoS", "DDoS",
        "Brute Force", "Brute Force", "Brute Force",
        "Brute Force", "Brute Force"
    ]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

print("Network Dataset:")
print(df)

# Step 3: Select input and output
X = df[["duration", "packets", "failed_logins"]]
y = df["label"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 6: Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 7: Test model
y_pred = model.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Step 9: Test new network traffic
new_data = pd.DataFrame({
    "duration": [15, 115, 215, 315],
    "packets": [30, 550, 850, 55],
    "failed_logins": [0, 0, 0, 15]
})

# Step 10: Predict attack types
predictions = model.predict(new_data)

print("\nCyber Attack Classification Results:")

for i in range(len(predictions)):
    print("Network", i+1, ":", predictions[i])
