from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from ml.autoencoder import AnomaliesAER
from ml.clusterization_and_knn import AnomaliesDetector
from ml.adtk_algos import AnomaliesADTK
from ml.isolation_forest import IsolationForestDetector


def find_anomalies_AER(timeseries):
    aer = AnomaliesAER(timeseries)
    anomalies = aer.detect_anomalies()
    return pd.DataFrame(anomalies)


def find_anomalies_kmeans(timeseries):
    detector = AnomaliesDetector(timeseries)
    kmeans_anomalies = detector.k_means()
    anomalies = kmeans_anomalies.loc[kmeans_anomalies.anomaly_cls == 1, ['datetime', 'value']]
    return anomalies.reset_index().drop(columns='index').rename(columns={'datetime': 'timestamp'})


def find_anomalies_knn(timeseries):
    detector = AnomaliesDetector(timeseries)
    knn_anomalies = detector.knn_anom()
    anomalies = knn_anomalies.loc[knn_anomalies['anomaly_knn'], ['datetime', 'value']]
    return anomalies.reset_index().drop(columns='index').rename(columns={'datetime': 'timestamp'})


def find_anomalies_iforest(timeseries):
    detector = IsolationForestDetector()
    detector.fit(timeseries)
    anomalies_df = detector.df[detector.df['anomaly'] == -1].drop(columns='anomaly').reset_index().drop(columns='index')
    return anomalies_df


def find_anomalies_adtk_kmeans(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_kmeans()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]
    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


def find_anomalies_adtk_outlier_detector(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_outliers()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]
    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


def find_anomalies_adtk_pca(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_pca()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]
    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


app = FastAPI(title="Anomaly Detection API",
              description="API for detecting anomalies in time series data",
              version="1.0")


@app.post("/detect_anomalies/{method}")
async def detect_anomalies(method: str, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    if method == 'aer':
        result = find_anomalies_AER(df)
    elif method == 'kmeans':
        result = find_anomalies_kmeans(df)
    elif method == 'knn':
        result = find_anomalies_knn(df)
    elif method == 'iforest':
        result = find_anomalies_iforest(df)
    elif method == 'adtk_kmeans':
        result = find_anomalies_adtk_kmeans(df)
    elif method == 'adtk_outlier':
        result = find_anomalies_adtk_outlier_detector(df)
    elif method == 'adtk_pca':
        result = find_anomalies_adtk_pca(df)
    else:
        raise HTTPException(status_code=400, detail="Invalid method")

    output = io.StringIO()
    result.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=anomalies_{method}.csv"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
