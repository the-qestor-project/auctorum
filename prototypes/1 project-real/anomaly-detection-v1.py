import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Power BI supplies the dataset variable automatically
df = dataset.copy()

# --- Robust handling: Use Calculated Profit if available, else Profit ---
profit_col = 'Calculated Profit' if 'Calculated Profit' in df.columns else 'Profit'

# Drop rows with missing profit or date
df = df.dropna(subset=[profit_col, 'Date'])

# Convert Date to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])

# Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[[profit_col]])

# Add anomaly score for further analysis
df['anomaly_score'] = model.decision_function(df[[profit_col]])

# Human-friendly flag
df['anomaly_flag'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# --- Visualization ---
colors = {'Anomaly': 'red', 'Normal': 'green'}
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df[profit_col], c=df['anomaly_flag'].map(colors), label=None)
plt.title('Anomaly Detection on Profit')
plt.xlabel('Date')
plt.ylabel(profit_col)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='green', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=8)
])
plt.show()

# Optionally, output anomaly summary for Power BI table visual
output = df[['Date', profit_col, 'anomaly_flag', 'anomaly_score']]
