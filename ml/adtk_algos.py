import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

from adtk.data import validate_series
from adtk.detector import MinClusterDetector, OutlierDetector
from adtk.detector import PcaAD


class AnomaliesADTK:
    def __init__(self, series):
        self.series = validate_series(series)

    def detect_outliers(self):
        outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.01))
        self.anomalies_outlier = outlier_detector.fit_detect(self.series)
        return self.anomalies_outlier

    def detect_kmeans(self):
        min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
        self.anomalies_clusterization = min_cluster_detector.fit_detect(self.series)
        return self.anomalies_clusterization

    def detect_pca(self):
        pca_ad = PcaAD(k=1)
        self.anomalies_pca = pca_ad.fit_detect(self.series)
        return self.anomalies_pca

    def plot_anomalies(self, anomalies, method):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.series.index, y=self.series['value'], mode='lines', name='Value'))

        anomaly_dates = anomalies[anomalies].index
        fig.add_trace(go.Scatter(x=anomaly_dates, y=self.series.loc[anomaly_dates]['value'],
                                 mode='markers', marker=dict(color='red'), name='Anomalies'))

        fig.update_layout(title=f'Anomalies in Time Series [ADTK {method}]',
                          xaxis_title='Datetime',
                          yaxis_title='Value')

        return fig
