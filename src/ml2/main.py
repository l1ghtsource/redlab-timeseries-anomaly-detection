from sklearn.ensemble import IsolationForest
import pandas as pd
from faststream import FastStream
from faststream.kafka import KafkaBroker
from time import time, sleep
from random import random
import clickhouse_connect


broker = KafkaBroker("kafka:9092", max_request_size=16000000)


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


default_client = clickhouse_connect.get_client(host='83.166.235.106', port=8123)
default_timeseries = default_client.query_df(
    'SELECT timestamp, web_response, throughput, apdex, error FROM "default"."test2" ORDER BY timestamp ASC')


@broker.subscriber("to_ml2")
@broker.publisher("from_ml2")
async def base_handler(body):
    if body['data_source'] == 'default':
        timeseries_all = default_timeseries
    else:
        host = body['data_source']['host']
        port = body['data_source']['port']
        query = body['data_source']['query']

        client = clickhouse_connect.get_client(host=host, port=port)
        timeseries_all = default_client.query_df(query)

    selected_value = body['column_name']
    timeseries = pd.concat([timeseries_all['timestamp'], timeseries_all[selected_value]], axis=1)
    timeseries.rename(columns={selected_value: 'value'}, inplace=True)

    detector = IsolationForestDetector()
    detector.fit(timeseries)

    anomalies = detector.df

    return {'time': time(), "msg_id": body['msg_id'], 'data': str(anomalies.to_csv())}

app = FastStream(broker,  description="Isolation Forest")
