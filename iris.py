import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("IRIS.CSV")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.describe())
print(df.tail())
print("\nMissing values in dataset:")
print(df.isnull().sum())
print("\nClass distribution:\n", df['species'].value_counts())

sns.pairplot(df, hue="species")
plt.show()

sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap="coolwarm")
plt.show()

sns.countplot(x='species', data=df)
plt.show()

# Encode target for multiclass classification
le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])

X = df.drop(columns=['species', 'target'])
y = df['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
