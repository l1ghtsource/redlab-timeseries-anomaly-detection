import pandas as pd
import plotly.graph_objects as go
from orion import Orion


def AER(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'].apply(lambda x: int(x.timestamp()))

    hyperparameters = {
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': 3600
        },
        'orion.primitives.aer.AER#1': {
            'epochs': 5,
            'verbose': True
        }
    }

    orion = Orion(
        pipeline='aer',
        hyperparameters=hyperparameters
    )

    orion.fit(data)

    anomalies = orion.detect(data)

    return anomalies


def AER_plot(data, anomalies):
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    anomalies['start'] = pd.to_datetime(anomalies['start'], unit='s')
    anomalies['end'] = pd.to_datetime(anomalies['end'], unit='s')

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['timestamp'],
                             y=data['value'],
                             mode='lines',
                             name='Time Series'))

    for _, row in anomalies.iterrows():
        anomaly_range = data[(data['timestamp'] >= row['start']) & (data['timestamp'] <= row['end'])]
        fig.add_trace(go.Scatter(x=anomaly_range['timestamp'],
                                 y=anomaly_range['value'],
                                 mode='lines',
                                 line=dict(color='red')))

    fig.update_layout(xaxis_title='Time',
                      yaxis_title='Value',
                      title='Anomalies in Time Series [AER]')
    return fig
