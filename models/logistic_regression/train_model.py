import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
try:
    df = pd.read_csv('cleaned_amazon_reviews.csv')
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    exit()
df['cleaned_text'] = df['cleaned_text'].fillna('')

X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    ))
])

parameters = {

    'tfidf__ngram_range': [(1, 1), (1, 2)],
    
    'tfidf__min_df': [2, 3, 5], 
    
    'clf__C': [0.1, 1, 10], 
}

grid_search = GridSearchCV(pipeline, parameters, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)

print("Starting GridSearchCV")
grid_search.fit(X_train, y_train)


print("\nGridSearch Complete.")
print(f"Best f1_macro score found: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)


print("\nEvaluating model on the test set...")
y_pred = grid_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\n--- Example Prediction ---")
new_text = ["this product is awesome and works perfectly"]
prediction = grid_search.predict(new_text)
print(f"Text: '{new_text[0]}'")
print(f"Predicted Sentiment: {prediction[0]}")

new_text_bad = ["waste of money, it broke immediately"]
prediction_bad = grid_search.predict(new_text_bad)
print(f"\nText: '{new_text_bad[0]}'")
print(f"Predicted Sentiment: {prediction_bad[0]}")

new_text_neutral = ["the product is okay, not great but not terrible"]
prediction_neutral = grid_search.predict(new_text_neutral)
print(f"\nText: '{new_text_neutral[0]}'")
print(f"Predicted Sentiment: {prediction_neutral[0]}")
# Define class names for better readability in the plot
class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix (Logistic Regression) - Raw Counts')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.show()
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix (Logistic Regression) - Normalized')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.show()
