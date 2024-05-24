import pandas as pd
import plotly.graph_objects as go
from orion import Orion


class AnomaliesAER:
    def __init__(self, data):
        self.data = data.copy()
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data['timestamp'] = self.data['timestamp'].apply(lambda x: int(x.timestamp()))

    def detect_anomalies(self, hyperparameters=None):
        if hyperparameters is None:
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

        orion.fit(self.data)

        self.anomalies = orion.detect(self.data)

        return self.anomalies

    def plot_anomalies(self):
        data_copy = self.data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'], unit='s')

        anomalies_copy = self.anomalies.copy()
        anomalies_copy['start'] = pd.to_datetime(anomalies_copy['start'], unit='s')
        anomalies_copy['end'] = pd.to_datetime(anomalies_copy['end'], unit='s')

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data_copy['timestamp'],
                                 y=data_copy['value'],
                                 mode='lines',
                                 name='Time Series'))

        for _, row in anomalies_copy.iterrows():
            anomaly_range = data_copy[(data_copy['timestamp'] >= row['start']) & (data_copy['timestamp'] <= row['end'])]
            fig.add_trace(go.Scatter(x=anomaly_range['timestamp'],
                                     y=anomaly_range['value'],
                                     mode='lines',
                                     line=dict(color='red')))

        fig.update_layout(xaxis_title='Time',
                          yaxis_title='Value',
                          title='Anomalies in Time Series [AER]')
        return fig
