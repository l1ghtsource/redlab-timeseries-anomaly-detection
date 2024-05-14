import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

from adtk.data import validate_series
from adtk.detector import MinClusterDetector, OutlierDetector
from adtk.detector import PcaAD


def ADTK_OutlierDetector(s):
    s = validate_series(s)
    outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.01))
    anomalies_outlier = outlier_detector.fit_detect(s)

    return anomalies_outlier


def ADTK_KMeans(s):
    s = validate_series(s)
    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
    anomalies_clusterization = min_cluster_detector.fit_detect(s)

    return anomalies_clusterization


def ADTK_Pca(s):
    s = validate_series(s)
    pca_ad = PcaAD(k=1)
    anomalies_pca = pca_ad.fit_detect(s)

    return anomalies_pca


def ADTK_plot(s, anomalies, method):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=s.index, y=s['value'], mode='lines', name='Value'))

    anomaly_dates = anomalies[anomalies].index
    fig.add_trace(go.Scatter(x=anomaly_dates, y=s.loc[anomaly_dates]['value'],
                             mode='markers', marker=dict(color='red'), name='Anomalies'))

    fig.update_layout(title=f'Anomalies in Time Series [ADTK {method}]',
                      xaxis_title='Datetime',
                      yaxis_title='Value')

    return fig
