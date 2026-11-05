
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create sample email dataset

data = {
    'sender': [
        'admin@gmail.com',
        'support@amazon.com',
        'info@microsoft.com',
        'security@google.com',
        'user@yahoo.com',
        'admin@amaz0n-security.com',
        'support@gmail.com',
        'verify@fake-bank.com',
        'alert@unknown-domain.com',
        'login@secure-account123.com',
        'service@paypal.com',
        'help@outlook.com',
        'update@paypa1-login.com',
        'security@fake-google.com',
        'customer@company.com'
    ],

    'reply_to': [
        'admin@gmail.com',
        'support@amazon.com',
        'info@microsoft.com',
        'security@google.com',
        'user@yahoo.com',
        'attacker@gmail.com',
        'attacker@yahoo.com',
        'hacker@gmail.com',
        'unknown@gmail.com',
        'hacker@yahoo.com',
        'service@paypal.com',
        'help@outlook.com',
        'attacker@gmail.com',
        'hacker@yahoo.com',
        'customer@company.com'
    ],

    'return_path': [
        'admin@gmail.com',
        'support@amazon.com',
        'info@microsoft.com',
        'security@google.com',
        'user@yahoo.com',
        'random@gmail.com',
        'random@yahoo.com',
        'hacker@gmail.com',
        'unknown@gmail.com',
        'hacker@yahoo.com',
        'service@paypal.com',
        'help@outlook.com',
        'attacker@gmail.com',
        'hacker@yahoo.com',
        'customer@company.com'
    ],

    'spf': [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 0, 0, 1
    ],

    'dkim': [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 0, 0, 1
    ],

    'dmarc': [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 0, 0, 1
    ],

    'label': [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 1, 1, 0
    ]
}

df = pd.DataFrame(data)

print("EMAIL DATASET:")
print(df)

# Step 2: Extract sender domains

df['sender_domain'] = df['sender'].apply(
    lambda x: x.split('@')[-1].lower()
)

df['reply_domain'] = df['reply_to'].apply(
    lambda x: x.split('@')[-1].lower()
)

df['return_domain'] = df['return_path'].apply(
    lambda x: x.split('@')[-1].lower()
)

# Step 3: Feature engineering

df['sender_length'] = df['sender'].apply(len)

df['domain_length'] = df['sender_domain'].apply(len)

df['domain_has_digits'] = df['sender_domain'].apply(
    lambda x: int(any(c.isdigit() for c in x))
)

df['reply_mismatch'] = (
    df['sender_domain'] != df['reply_domain']
).astype(int)

df['return_mismatch'] = (
    df['sender_domain'] != df['return_domain']
).astype(int)

df['special_characters'] = df['sender'].apply(
    lambda x: sum(not c.isalnum() for c in x)
)

# Step 4: Select machine learning features

features = [
    'spf',
    'dkim',
    'dmarc',
    'sender_length',
    'domain_length',
    'domain_has_digits',
    'reply_mismatch',
    'return_mismatch',
    'special_characters'
]

X = df[features]
y = df['label']

print("\nEXTRACTED FEATURES:")
print(X)

# Step 5: Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Step 6: Train Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")

# Step 7: Predict email spoofing

y_pred = model.predict(X_test)

# Step 8: Evaluate model

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION:")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Step 9: Display email detection results

results = df.loc[X_test.index, ['sender']].copy()

results['Actual'] = y_test.map({
    0: 'Legitimate',
    1: 'Spoofed'
})

results['Predicted'] = [
    'Spoofed' if p == 1 else 'Legitimate'
    for p in y_pred
]

print("\nEMAIL SPOOFING DETECTION RESULTS:")
print(results.to_string(index=False))

# Step 10: Test a new email

new_email = pd.DataFrame([{
    'spf': 0,
    'dkim': 0,
    'dmarc': 0,
    'sender_length': len('admin@fake-bank123.com'),
    'domain_length': len('fake-bank123.com'),
    'domain_has_digits': 1,
    'reply_mismatch': 1,
    'return_mismatch': 1,
    'special_characters': 1
}])

prediction = model.predict(new_email)[0]

print("\nNEW EMAIL ANALYSIS:")

if prediction == 1:
    print("Result: Suspicious / Potentially Spoofed Email")
else:
    print("Result: Legitimate Email")

# Step 11: Feature importance

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(importance.to_string(index=False))
