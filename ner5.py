# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Load input features and output labels
X = np.loadtxt('vectorized.csv', delimiter=',', skiprows=1)
y = np.loadtxt('output_file.csv', delimiter=';', skiprows=1, usecols=(4,), dtype=int)

# Print shapes to verify lengths
print("X shape:", X.shape)  # Should be (n_samples, 100)
print("y shape:", y.shape)  # Should be (n_samples,)

# Ensure consistent length
if len(X) > len(y):
    X = X[:len(y)]
elif len(X) < len(y):
    y = y[:len(X)]

# Standardize the features
X = preprocessing.StandardScaler().fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model for multiclass classification
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1.0)
# model = KNeighborsClassifier(n_neighbors=5)
model = RandomForestClassifier(n_estimators=100, random_state=42)  

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# for i in range(len(y_pred)):
#     print(y_pred[i], y_test[i]) 
# print(classification_report(y_test, y_pred))

