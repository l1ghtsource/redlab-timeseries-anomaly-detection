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
                    'interval': 300
                },
                'orion.primitives.aer.AER#1': {
                    'epochs': 1,
                    'verbose': True
                }
            }

        orion = Orion(
            pipeline='aer',
            hyperparameters=hyperparameters
        )

        orion.fit(self.data[int(0.6*len(self.data)):int(0.7*len(self.data))])

        self.anomalies = orion.detect(self.data)

        return self.anomalies
