import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

 #  THEME #
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = '#f8fafc'
plt.rcParams['axes.facecolor'] = '#ffffff'


# LOAD DATA #
df = pd.read_csv("C:/Users/91898/OneDrive/Desktop/GRAPHS AND PYTHON/non_iou_zipcodes_2024.csv")


#  CLEAN DATA
df = df[(df['res_rate'] > 0) & (df['comm_rate'] > 0) & (df['ind_rate'] > 0)]

df['res_rate'] *= 100
df['comm_rate'] *= 100
df['ind_rate'] *= 100

df = df.drop_duplicates()

print("Shape:", df.shape)

# BASIC EDA #

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIPTION ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# DISTRIBUTION #
plt.figure(figsize=(10,5))
sns.histplot(df['res_rate'], kde=True, color='#ff6b6b')
plt.title("Residential Rate Distribution")
plt.show()

#  OUTLIERS #
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['res_rate','comm_rate','ind_rate']])
plt.title("Outlier Detection")
plt.show()

#  RELATIONSHIP #

sns.pairplot(df[['res_rate','comm_rate','ind_rate']])
plt.show()

#  HEATMAP #
plt.figure(figsize=(6,4))
sns.heatmap(df[['res_rate','comm_rate','ind_rate']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#  CATEGORY ANALYSIS #
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='ownership',
              hue='ownership', palette='Set2', legend=False)
plt.title("Ownership Distribution")
plt.xticks(rotation=20)
plt.show()

# Avg rates #
ownership_avg = df.groupby('ownership')[['res_rate','comm_rate','ind_rate']].mean()

ownership_avg.plot(kind='bar', figsize=(10,5),
                   color=['#ff9f1c','#2ec4b6','#e71d36'])
plt.title("Avg Rates by Ownership")
plt.xticks(rotation=20)
plt.show()

# STATE ANALYSIS #
top_states = df.groupby('state')['res_rate'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_states.plot(kind='bar', color='#6a5acd')
plt.title("Top 10 States by Residential Rate")
plt.xticks(rotation=45)
plt.show()

#  MACHINE LEARNING

le = LabelEncoder()
df['ownership_enc'] = le.fit_transform(df['ownership'])
df['state_enc'] = le.fit_transform(df['state'])

X = df[['comm_rate','ind_rate','ownership_enc','state_enc']]
y = df['res_rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#  RESULTS #
print("\nR2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Predictions Table
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}).round(2)

print("\nActual vs Predicted:\n")
print(results.head(20))

# MODEL VISUALIZATION #
plt.figure(figsize=(12,5))

# SCATTER PLOT #
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred, color='#4ecdc4')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.title("Actual vs Predicted")

# LINE PLOT #
plt.subplot(1,2,2)
plt.plot(y_test.values[:20], label='Actual', marker='o')
plt.plot(y_pred[:20], label='Predicted', marker='x')
plt.legend()
plt.title("First 20 Comparison")

plt.tight_layout()
plt.show()
