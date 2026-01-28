from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (toy example)
X = [[150, 0], [170, 1], [160, 0], [180, 1]]
y = [0, 1, 0, 1]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model and print accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
