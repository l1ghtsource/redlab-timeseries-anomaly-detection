import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

SEED = sum(ord(ch) for ch in 'MISISFOUNDHACK')


class IsolationForestDetector:
    def __init__(self, contamination=0.001, n_estimators=200, max_samples=0.7, random_state=SEED):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination,
                                     n_estimators=n_estimators,
                                     max_samples=max_samples,
                                     random_state=random_state)

    def fit(self, df):
        X = df[['value']].values
        self.model.fit(X)
        df['anomaly'] = self.model.predict(X)
        self.df = df
        return df

    def plot(self):
        if not hasattr(self, 'df'):
            raise ValueError("No fit data found. Please run fit() first.")

        s = self.df

        anomalies = s['anomaly'] == -1

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=s.index, y=s['value'], mode='lines', name='Value'))

        anomaly_dates = s[anomalies].index
        fig.add_trace(go.Scatter(x=anomaly_dates, y=s.loc[anomaly_dates]['value'],
                                 mode='markers', marker=dict(color='red'), name='Anomalies'))

        fig.update_layout(title=f'Anomalies in Time Series [Isolation Forest]',
                          xaxis_title='Datetime',
                          yaxis_title='Value')

        return fig
