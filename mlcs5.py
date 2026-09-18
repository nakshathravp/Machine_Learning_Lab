import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "url": [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "http://secure-login-example.com",
        "http://account-verify-example.com",
        "http://free-money-example.com/login",
        "http://192.168.1.10/login"
    ],
    "label": [
        0, 0, 0, 0, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

def extract_features(url):
    parsed = urlparse(url)

    features = [
        len(url),                         
        url.count("."),                  
        url.count("-"),                   
        url.count("/"),                   
        url.count("@"),                   
        sum(c.isdigit() for c in url),   
        int(parsed.scheme == "https"),    
        len(parsed.netloc.split("."))     
    ]

    return features

X = df["url"].apply(extract_features).tolist()
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_url = "http://suspicious-login-example.com"
new_features = [extract_features(new_url)]
new_features = scaler.transform(new_features)

prediction = model.predict(new_features)

if prediction[0] == 0:
    print("\nNew URL: Legitimate")
else:
    print("\nNew URL: Malicious")