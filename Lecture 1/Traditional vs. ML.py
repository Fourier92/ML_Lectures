import random

# Rule-Based Approach to Distinguish Cats from Dogs
def classify_animal_rule_based(features):
    """
    A simple rule-based approach to classify an animal as a cat or a dog.
    features: dict containing characteristics like 'weight', 'sound', 'fur_length'
    """
    if features['sound'] == 'meow':
        return 'Cat'
    elif features['sound'] == 'bark':
        return 'Dog'
    elif features['weight'] < 10 and features['fur_length'] < 5:
        return 'Cat'
    elif features['weight'] > 10 and features['fur_length'] >= 5:
        return 'Dog'
    else:
        return 'Unknown'

# Example usage
random_animal = {'weight': 8, 'sound': 'meow', 'fur_length': 3}
print("Rule-Based Classification:", classify_animal_rule_based(random_animal))

# Machine Learning Approach to Classify Cats and Dogs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Generate sample dataset (features: weight, fur_length; labels: 0=Cat, 1=Dog)
data = np.array([
    [4, 2, 0], [5, 3, 0], [6, 3, 0], [8, 4, 0], # Cats
    [12, 5, 1], [15, 6, 1], [20, 7, 1], [25, 8, 1]  # Dogs
])

X = data[:, :2]  # Features: weight and fur_length
y = data[:, 2]   # Labels: 0 for Cat, 1 for Dog

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Predict on a new sample (weight=9, fur_length=3)
new_sample = np.array([[9, 3]])
prediction = model.predict(new_sample)
print("ML-Based Classification:", 'Dog' if prediction[0] == 1 else 'Cat')
