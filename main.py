# =========================================
# 🎓 Student Performance Prediction System
# (Final Version with Model Saving)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# 🎨 Style Settings
# =========================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# =========================================
# 📁 Create Required Folders
# =========================================
folders = ["data", "outputs", "models"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =========================================
# 📊 Step 1: Create Synthetic Dataset
# =========================================
np.random.seed(42)
data_size = 200

data = pd.DataFrame({
    'study_hours': np.random.randint(1, 10, data_size),
    'attendance': np.random.randint(50, 100, data_size),
    'previous_marks': np.random.randint(40, 100, data_size),
    'assignments_completed': np.random.randint(1, 10, data_size)
})

data['final_score'] = (
    data['study_hours'] * 5 +
    data['attendance'] * 0.3 +
    data['previous_marks'] * 0.4 +
    data['assignments_completed'] * 2 +
    np.random.randn(data_size) * 5
)

data.to_csv('data/student_data.csv', index=False)
print("✅ Dataset created successfully!")

# =========================================
# 📊 EDA VISUALS
# =========================================

# Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True, cmap='Blues', linewidths=0.5)
plt.title("Correlation Between Features")
plt.tight_layout()
plt.savefig("outputs/heatmap.png")
plt.close()

# Distribution
plt.figure()
sns.histplot(data['final_score'], kde=True, bins=20)
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/distribution.png")
plt.close()

# Study vs Score
plt.figure()
sns.regplot(x=data['study_hours'], y=data['final_score'])
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig("outputs/study_vs_score.png")
plt.close()

# =========================================
# 🤖 Step 2: Model Training
# =========================================
X = data.drop('final_score', axis=1)
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ✅ SAVE MODEL HERE
joblib.dump(model, "models/student_model.pkl")
print("✅ Model saved in models/student_model.pkl")

# =========================================
# 🎯 Step 3: Prediction
# =========================================
y_pred = model.predict(X_test)

# =========================================
# 📈 Step 4: Evaluation
# =========================================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# =========================================
# 📊 Advanced Visuals
# =========================================

# Actual vs Predicted
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Performance")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Feature Importance
importance = model.coef_
features = X.columns

plt.figure()
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Impact on Final Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("📁 All graphs saved in 'outputs' folder")

# =========================================
# 🎯 Step 5: New Prediction
# =========================================
new_student = pd.DataFrame({
    'study_hours': [6],
    'attendance': [85],
    'previous_marks': [70],
    'assignments_completed': [7]
})

prediction = model.predict(new_student)
print("\n🎯 Predicted Score for New Student:", round(prediction[0], 2))