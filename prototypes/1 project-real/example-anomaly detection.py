import pandas as pd
from sklearn.ensemble import IsolationForest

# dataset is provided by Power BI
df = dataset.copy()
model = IsolationForest(contamination=0.05)
df['anomaly'] = model.fit_predict(df[['Profit']])
df['anomaly_flag'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

import matplotlib.pyplot as plt
colors = {'Anomaly':'red', 'Normal':'green'}
plt.scatter(df['Date'], df['Profit'], c=df['anomaly_flag'].map(colors))
plt.title('Profit Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.show()
