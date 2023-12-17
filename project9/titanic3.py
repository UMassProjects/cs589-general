# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Load data
data = pd.read_csv("train.csv")

# Preprocessing
# Separate features and target
data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
features = data.drop("Survived", axis=1)
target = data["Survived"]


# Impute missing values
features = features.fillna(features.median(axis=0))

# Scale numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.select_dtypes(include=["float64", "int64"]))

# Encode categorical features
encoder = OneHotEncoder(sparse=False)
features_encoded = encoder.fit_transform(features.select_dtypes(include=["object"]))

# Combine features
features_combined = np.concatenate([features_scaled, features_encoded], axis=1)

# Dimensionality reduction with PCA-sphering
# pca = PCA(n_components=0.95, whiten=True)
# features_transformed = pca.fit_transform(features_combined)

features_transformed = features_combined

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_transformed, target, test_size=0.2, random_state=42)

# Model implementations and evaluation

# SVM
model_svm = SVC(kernel="rbf")

# Hyperparameter tuning
param_grid_svm = {
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1],
}

from sklearn.model_selection import GridSearchCV

clf_svm = GridSearchCV(model_svm, param_grid_svm, scoring="f1")
clf_svm.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)

print("SVM F1 score:", f1_score(y_test, y_pred_svm))

# MLP
model_mlp = MLPClassifier()

# Hyperparameter tuning
param_grid_mlp = {
    "hidden_layer_sizes": [(100,), (50, 50), (100, 50)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "sgd"],
}

clf_mlp = GridSearchCV(model_mlp, param_grid_mlp, scoring="roc_auc")
clf_mlp.fit(X_train, y_train)

y_pred_mlp = clf_mlp.predict(X_test)

print("MLP AUC score:", roc_auc_score(y_test, y_pred_mlp))

# Random Forest
model_rf = RandomForestClassifier()

# Hyperparameter tuning
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "max_features": ["auto", "sqrt", "log2"],
}

clf_rf = GridSearchCV(model_rf, param_grid_rf, scoring="accuracy")
clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)

print("Random Forest accuracy:", accuracy_score(y_test, y_pred_rf))

# Further analysis and report generation
# ...

