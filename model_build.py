import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- 1. Advanced Synthetic Data Generation (Statistical Approach) ---
# Hum 'If-Else' nahi, balki 'Probability Distributions' use karenge.
# Yeh industry standard hai jab real data confidential hota hai.

np.random.seed(42) # Result har baar same aayega (Reproducible)
n_samples = 2000   # 2000 Flights ka data

# Base Features
data = {
    'Airline': np.random.randint(0, 5, n_samples),       # 0=Delta, 1=American, etc.
    'Month': np.random.randint(1, 13, n_samples),        # Jan-Dec
    'Distance': np.random.randint(100, 3000, n_samples), # Miles
    'DepTime': np.random.randint(500, 2400, n_samples)   # 5:00 AM to 12:00 PM
}
df = pd.DataFrame(data)

# --- 2. Creating Mathematical Correlations (Patterns for ML to Learn) ---

# Pattern 1: Late Night Flights (Time > 6 PM) have higher delay probability due to traffic
# Pattern 2: Longer Distance = Slightly higher risk
# Pattern 3: Winter Months (1, 2, 12) have weather delays

# Hum ek "Risk Score" calculate karenge
risk_score = (
    (df['DepTime'] / 2400) * 0.5 +        # Time ka 50% impact
    (df['Distance'] / 3000) * 0.3 +       # Distance ka 30% impact
    (np.isin(df['Month'], [1, 2, 12]).astype(int)) * 0.4 # Winter ka impact
)

# Thoda randomness (Noise) add karte hain taaki model ko 'ratta' na maarna pade
noise = np.random.normal(0, 0.2, n_samples)
final_risk = risk_score + noise

# Threshold set karte hain (Top 30% flights delay hongi)
threshold = np.percentile(final_risk, 70)
df['Delay'] = (final_risk > threshold).astype(int)

# --- 3. Professional ML Pipeline ---

X = df[['Airline', 'Month', 'Distance', 'DepTime']]
y = df['Delay']

# Split Data (80% Train, 20% Test) - Standard Practice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Model (Real Random Forest)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Training
print("🚀 Training Model on Synthetic Pattern Data...")
model.fit(X_train, y_train)

# Evaluation (Testing the intelligence of model)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance Report:")
print(f"--------------------------")
print(f"✅ Accuracy: {accuracy*100:.2f}%")
print(f"\n📝 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --- 4. Save the Logic ---
pickle.dump(model, open('xgbmodel.pkl', 'wb'))
print("\n Professional Model Saved as 'xgbmodel.pkl'")