from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import zipfile
import clickhouse_connect

from ml.isolation_forest import IsolationForestDetector


def download_zip(data):
    zip_buffer = io.BytesIO()

    dataframes = [x[0] for x in data]
    names = [x[1] for x in data]

    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for index, df in enumerate(dataframes):
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            zip_file.writestr(f'{names[index]}_anomalies.csv', csv_buffer.read())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def find_anomalies_iforest(timeseries):
    detector = IsolationForestDetector()
    detector.fit(timeseries)

    return detector.df


async def detect_anomalies(start: str, end: str):
    try:
        client = clickhouse_connect.get_client(host='83.166.235.106', port=8123)
        timeseries = client.query_df(
            f'SELECT timestamp as time, web_response, throughput, apdex, error FROM "default"."test2" ORDER BY time ASC')
        timeseries.rename(columns={'time': 'timestamp'}, inplace=True)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось извлечь данные из БД: {e}")

    results = []

    for _, column in enumerate(timeseries.columns[1:]):
        ts = timeseries[['timestamp', column]]
        ts.rename(columns={column: 'value'}, inplace=True)

        df = find_anomalies_iforest(ts)
        df = df[(pd.to_datetime(df['timestamp']) >= start) & (pd.to_datetime(df['timestamp']) <= end)]

        anomalies_df = df[df['anomaly'] == -1].drop(columns='anomaly').reset_index().drop(columns='index')

        results.append([anomalies_df, column])

    zip_content = download_zip(results)

    return zip_content


app = FastAPI(title="Anomaly Detection API",
              description="API for detecting anomalies in time series data",
              version="1.0")


@app.post("/detect_anomalies/", response_class=Response)
async def detect_anomalies_api(
    start_date: str,
    end_date: str
):
    zip_content = await detect_anomalies(start_date, end_date)

    return Response(content=zip_content, media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=anomalies.zip"})


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
