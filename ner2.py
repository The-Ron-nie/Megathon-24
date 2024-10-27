# Import necessary libraries
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
X = np.loadtxt('vectorized.csv', delimiter=',', skiprows=1)
y = np.loadtxt('mental_health_dataset.csv', delimiter=',', skiprows=1, usecols=(3,), dtype=str)

# Print shapes to verify lengths
print("X shape:", X.shape)
print("y shape:", y.shape)

# Ensure consistent length
if len(X) > len(y):
    X = X[:len(y)]
elif len(X) < len(y):
    y = y[:len(X)]

yt = np.zeros(len(y))
dict = {
    "Health Anxiety": 1,
    "Eating Disorder": 2, 
    "Anxiety": 3,
    "Depression": 4,
    "Insomnia": 5,
    "Stress": 6,
    "Positive Outlook": 7,
    "Career Confusion": 8
}
for i in range(len(y)):
    if y[i] in dict:
        yt[i] = dict[y[i]]
y = yt
# X = preprocessing.StandardScaler().fit_transform(X)
X = preprocessing.StandardScaler().fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM model
# model = SVC(kernel='linear', C=1.0)  # You can choose different kernels like 'linear', 'rbf', etc.

# Create a KNN model
# model = KNeighborsClassifier(n_neighbors=5)
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