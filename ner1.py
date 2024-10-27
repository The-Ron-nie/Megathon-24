from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
import numpy as np
# Sample data (replace with your actual data)
# X = np.random.randn(1000, 100)  # 1000 samples, 100 features
X = np.loadtxt('vectorized.csv', delimiter=',', skiprows=1)
X = X[:,:]
# y = np.random.choice([-1, 0, 1], 1000)  # 1000 labels
y = np.loadtxt('mental_health_dataset.csv', delimiter=',', skiprows=1, usecols=(1,), dtype=str)
if len(X) > len(y):
    X = X[:len(y)]
elif len(X) < len(y):
    y = y[:len(X)]
yt = np.zeros(len(y))
for i in range(len(y)):
    if y[i] == "Positive":
        yt[i] = 1
    elif y[i] == "Negative":
        yt[i] = -1
    else:
        yt[i] = 0
y = yt

X = preprocessing.StandardScaler().fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1.0)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# print(classification_report(y_test, y_pred))
ans=model.predict("I am feeling very sad today")
print(ans)