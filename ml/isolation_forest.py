import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

SEED = sum(ord(ch) for ch in 'MISISFOUNDHACK')


def IF(df, contamination=0.005, n_estimators=200, max_samples=0.7, random_state=SEED):
    X = df[['value']].values

    model = IsolationForest(contamination=contamination,
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            random_state=random_state)
    model.fit(X)

    df['anomaly'] = model.predict(X)

    return df


def IF_plot(s, anomalies):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=s.index, y=s['value'], mode='lines', name='Value'))

    anomaly_dates = anomalies[anomalies].index
    fig.add_trace(go.Scatter(x=anomaly_dates, y=s.loc[anomaly_dates]['value'],
                             mode='markers', marker=dict(color='red'), name='Anomalies'))

    fig.update_layout(title=f'Anomalies in Time Series [Isolation Forest]',
                      xaxis_title='Datetime',
                      yaxis_title='Value')

    return fig
