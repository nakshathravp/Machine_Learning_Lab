
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import ipaddress

# Sample wireless sensor network data
data = {
    "timestamp": [10, 11, 12, 13, 14, 90, 95, 98, 100, 105],

    "source_ip": [
        "192.168.1.10",
        "192.168.1.11",
        "192.168.1.12",
        "192.168.1.13",
        "192.168.1.14",
        "10.0.0.10",
        "10.0.0.11",
        "10.0.0.12",
        "10.0.0.13",
        "10.0.0.14"
    ],

    "event_type": [
        "DATA", "DATA", "DATA", "LOGIN", "DATA",
        "FLOOD", "FLOOD", "SCAN", "FLOOD", "SCAN"
    ],

    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Create dataset
df = pd.DataFrame(data)

# Convert IP addresses into numbers
df["source_ip"] = df["source_ip"].apply(
    lambda ip: int(ipaddress.ip_address(ip))
)

# Convert event types into numbers
event_map = {
    "DATA": 0,
    "LOGIN": 1,
    "FLOOD": 2,
    "SCAN": 3
}

df["event_type"] = df["event_type"].map(event_map)

# Select input features
X = df[["timestamp", "source_ip", "event_type"]]

# Select output labels
y = df["label"]

# Create machine learning model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X, y)

print("Model trained successfully!")

# Test new network events
test_data = pd.DataFrame({
    "timestamp": [15, 110],
    "source_ip": ["192.168.1.20", "10.0.0.20"],
    "event_type": ["DATA", "FLOOD"]
})

# Convert test IP addresses into numbers
test_data["source_ip"] = test_data["source_ip"].apply(
    lambda ip: int(ipaddress.ip_address(ip))
)

# Convert test event types into numbers
test_data["event_type"] = test_data["event_type"].map(event_map)

# Predict events
prediction = model.predict(test_data)

# Display results
for i in range(len(prediction)):

    if prediction[i] == 0:
        print("Event", i+1, ": Normal Event")

    else:
        print("Event", i+1, ": Malicious Event")
