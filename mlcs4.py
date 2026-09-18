import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = {
    "url": [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.amazon.com/login",
        "https://github.com/openai",
        "https://www.microsoft.com/en-us/",
        "https://www.paypal.com/home",
        "https://www.bankofamerica.com/security",
        "https://www.linkedin.com/in/johndoe",
        "https://www.apple.com/store",

        # Phishing URLs
        "http://paypal-login-security-alert.com",
        "http://verify-account-amazon.biz/login",
        "https://secure-facebook-login-update.ru",
        "http://bankofamerica-login-confirm.net",
        "https://google-account-reset-now.xyz",
        "http://microsoft-support-verification.top",
        "http://appleid-reset-alert.org",
        "https://login-update-github.io"
    ],

    "label": [
        0, 0, 0, 0, 0, 0, 0, 0,   # 0 = Legitimate
        1, 1, 1, 1, 1, 1, 1, 1    # 1 = Phishing
    ]
}

df = pd.DataFrame(data)

print("\nDataset created successfully.")
print("\nDataset:")
print(df)

X = df["url"]
y = df["label"]

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_vectorized = vectorizer.fit_transform(X)

print("\nTF-IDF feature extraction completed.")
print("Feature matrix shape:", X_vectorized.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)

plt.title("Confusion Matrix - URL Phishing Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

joblib.dump(model, "phishing_url_model.pkl")
joblib.dump(vectorizer, "phishing_vectorizer.pkl")

print("\nModel saved as: phishing_url_model.pkl")
print("Vectorizer saved as: phishing_vectorizer.pkl")

new_urls = [
    "https://secure-login-paypal-alert.info",
    "https://www.amazon.in/gp/cart/view.html",
    "http://update-facebook-verification.com",
    "https://accounts.google.com"
]

new_features = vectorizer.transform(new_urls)

predictions = model.predict(new_features)

for url, prediction in zip(new_urls, predictions):

    if prediction == 1:
        status = "Phishing"
    else:
        status = "Legitimate"

    print(f"\n{url}")
    print(f"Prediction: {status}")

