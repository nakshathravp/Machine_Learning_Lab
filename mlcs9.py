
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create email dataset

data = {
    'email': [
        "Win a free prize now",
        "Congratulations you won a lottery",
        "Claim your free gift today",
        "Get a free offer now",
        "You have won a cash prize",
        "Free entry in lucky draw",
        "Win money by clicking here",
        "Limited time special offer",
        "Claim your reward now",
        "Get free shopping vouchers",
        "Meeting scheduled for tomorrow",
        "Please submit your assignment",
        "Project discussion at 10 AM",
        "Your class starts at 9 AM",
        "Please check your email",
        "Team meeting is postponed",
        "Submit the report today",
        "Your appointment is confirmed",
        "Please attend the online meeting",
        "The project deadline is Monday",
        "Congratulations claim your bonus",
        "Free cash reward waiting for you",
        "You are selected for a free gift",
        "Click here to win money",
        "Exclusive discount offer today",
        "Please review the project report",
        "Meeting notes are attached",
        "Your exam schedule is updated",
        "Send me the assignment details",
        "Thank you for your response"
    ],

    'label': [
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'not spam', 'not spam', 'not spam', 'not spam', 'not spam',
        'not spam', 'not spam', 'not spam', 'not spam', 'not spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'not spam', 'not spam', 'not spam', 'not spam', 'not spam'
    ]
}

df = pd.DataFrame(data)

print("EMAIL DATASET:")
print(df)

# Step 2: Split data

X = df['email']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Convert text into numerical features

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes classifier

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Step 5: Predict test emails

y_pred = model.predict(X_test)

# Step 6: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy * 100, "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 7: Test new emails

new_emails = [
    "Congratulations you won a free prize",
    "Please attend the project meeting tomorrow",
    "Click here to claim your cash reward",
    "Please submit your assignment today"
]

new_data = vectorizer.transform(new_emails)

predictions = model.predict(new_data)

print("\nEMAIL CLASSIFICATION RESULTS:")

for email, result in zip(new_emails, predictions):
    print("\nEmail:", email)
    print("Category:", result.upper())

print("\nSpam Detection Completed Successfully!")
