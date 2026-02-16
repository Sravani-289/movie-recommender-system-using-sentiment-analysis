import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# sample movie reviews dataset
data = {
    'review': [
        "This movie is amazing and fantastic",
        "Worst movie ever",
        "I loved the acting and story",
        "Very boring and slow movie",
        "Excellent direction and screenplay",
        "Not worth watching"
    ],
    'sentiment': ['positive','negative','positive','negative','positive','negative']
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test prediction
while True:
    user_review = input("Enter movie review: ")
    user_vector = vectorizer.transform([user_review])
    prediction = model.predict(user_vector)
    print("Predicted Sentiment:", prediction[0])
