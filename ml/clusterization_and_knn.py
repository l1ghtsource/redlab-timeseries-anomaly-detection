import pandas as pd
import numpy as np
from numpy import percentile

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go

SEED = sum(ord(ch) for ch in 'MISISFOUNDHACK')


def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0, len(data)):
        X_a = np.array(data.loc[i])
        X_b = model.cluster_centers_[model.labels_[i] - 1]
        distance.at[i] = np.linalg.norm(X_a - X_b)
    return distance


def K_Means(data):
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns='timestamp')
    data = data.set_index('datetime')

    dd = pd.DataFrame(data['value'].resample('D').sum())
    dd['prev'] = dd['value'].shift(periods=1)
    dd['prev_7'] = dd['value'].shift(periods=7)
    dd.dropna(inplace=True)

    X = dd.values
    X_std = StandardScaler().fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    data_new = pd.DataFrame(X_std)

    pca = PCA(n_components=2, random_state=SEED)
    data_new = pca.fit_transform(data_new)

    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data_new)
    data_new = pd.DataFrame(np_scaled)

    kmeans_model = KMeans(n_clusters=7, random_state=42 * SEED).fit(data_new)

    dd.reset_index(inplace=True)
    dd['cluster'] = kmeans_model.predict(data_new)
    dd.index = data_new.index
    dd['pca1'] = data_new[0]
    dd['pca2'] = data_new[1]

    dist = getDistanceByPoint(data_new, kmeans_model)
    threshold = percentile(dist, 95)
    dd['anomaly_cls'] = (dist >= threshold).astype(int)

    return dd


def KNN_Anom(data):
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns='timestamp')
    data = data.set_index('datetime')

    dd_ = pd.DataFrame(data['value'].resample('D').sum())
    dd_['prev'] = dd_['value'].shift(periods=1)
    dd_['prev_7'] = dd_['value'].shift(periods=7)
    dd_.dropna(inplace=True)

    X = dd_.values
    X_std = StandardScaler().fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    data_new = pd.DataFrame(X_std)

    pca = PCA(n_components=2, random_state=SEED)
    data_new = pca.fit_transform(data_new)

    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data_new)
    data_new = pd.DataFrame(np_scaled)

    kmeans_model = KMeans(n_clusters=7, random_state=42*SEED).fit(data_new)

    dd_.reset_index(inplace=True)
    dd_['cluster'] = kmeans_model.predict(data_new)
    dd_.index = data_new.index
    dd_['pca1'] = data_new[0]
    dd_['pca2'] = data_new[1]

    dist = getDistanceByPoint(data_new, kmeans_model)
    threshold = percentile(dist, 95)
    dd_['anomaly_cls'] = (dist >= threshold).astype(int)

    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(X)

    dist, _ = knn.kneighbors(X)

    dd_['dist'] = dist.mean(axis=1)
    threshold = percentile(dd_.dist, 95)

    dd_['anomaly_knn'] = dd_.dist > threshold

    return dd_


def KMeans_plot(dd):
    a = dd.loc[dd.anomaly_cls == 1, ['value', 'datetime']]

    scatter_trace = go.Scatter(x=dd['datetime'],
                               y=dd['value'],
                               mode='lines',
                               name='All Data')

    anomaly_trace = go.Scatter(x=a['datetime'],
                               y=a['value'],
                               mode='markers',
                               marker=dict(color='red'),
                               name='Anomalies')

    layout = go.Layout(
        title='Anomalies in Time Series [K-MEANS]',
        xaxis=dict(title='Datetime'),
        yaxis=dict(title='Value'),
        height=500
    )

    fig = go.Figure(data=[scatter_trace, anomaly_trace], layout=layout)

    return fig


def KNN_plot(dd_):
    a = dd_.loc[dd_['anomaly_knn'], ['value', 'datetime']]

    scatter_trace = go.Scatter(x=dd_['datetime'],
                               y=dd_['value'],
                               mode='lines',
                               name='All Data')

    anomaly_trace = go.Scatter(x=a['datetime'],
                               y=a['value'],
                               mode='markers',
                               marker=dict(color='red'),
                               name='Anomalies')

    layout = go.Layout(
        title='Anomalies in Time Series [KNN]',
        xaxis=dict(title='Datetime'),
        yaxis=dict(title='Value'),
        height=500
    )

    fig = go.Figure(data=[scatter_trace, anomaly_trace], layout=layout)

    return fig
