import pandas as pd
from sklearn.ensemble import IsolationForest

urls = [
    "https://google.com",
    "https://www.amazon.com",
    "https://github.com",
    "https://microsoft.com",
    "https://wikipedia.org",
    "http://example.com/login",
    "http://secure-login-account.com",
    "http://verify-user-account123.com/login",
    "http://192.168.1.10/login.php",
    "http://free-money-win-prize123456.com"
]

def extract_features(url):
    return [
        len(url),                         # URL length
        url.count("."),                   # Number of dots
        url.count("-"),                   # Number of hyphens
        sum(c.isdigit() for c in url),    # Number of digits
        sum(not c.isalnum() for c in url),# Special characters
        int(url.startswith("https"))      # HTTPS
    ]

features = [extract_features(url) for url in urls]

df = pd.DataFrame(
    features,
    columns=[
        "url_length",
        "dots",
        "hyphens",
        "digits",
        "special_chars",
        "https"
    ]
)

print("Extracted Features:")
print(df)

model = IsolationForest(
    contamination=0.3,
    random_state=42
)

model.fit(df)

df["prediction"] = model.predict(df)

df["result"] = df["prediction"].map({
    1: "Normal",
    -1: "Anomalous"
})

df.insert(0, "URL", urls)

print("\nURL Anomaly Detection Results:")
print(df[["URL", "result"]])