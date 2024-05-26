from orion import Orion
import pandas as pd
from faststream import FastStream
from faststream.kafka import KafkaBroker
from time import time, sleep
from random import random
import clickhouse_connect


broker = KafkaBroker("kafka:9092", max_request_size=16000000)


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


default_client = clickhouse_connect.get_client(host='83.166.235.106', port=8123)
default_timeseries = default_client.query_df(
    'SELECT timestamp, web_response, throughput, apdex, error FROM "default"."test2" ORDER BY timestamp ASC')
# default_timeseries.rename(columns={'time': 'timestamp'}, inplace=True)


@broker.subscriber("to_ml1")
@broker.publisher("from_ml1")
async def base_handler(body):
    if body['data_source'] == 'default':
        timeseries_all = default_timeseries
    else:
        host = body['data_source']['host']
        port = body['data_source']['port']
        query = body['data_source']['query']

        client = clickhouse_connect.get_client(host=host, port=port)
        timeseries_all = default_client.query_df(query)
        # timeseries_all.rename(columns={timestamp_column: 'timestamp'}, inplace=True)

    # numeric_columns = timeseries_all.select_dtypes(include=['float', 'int'])
    # numeric_column_names = numeric_columns.columns.tolist()

    selected_value = body['column_name']
    timeseries = pd.concat([timeseries_all['timestamp'], timeseries_all[selected_value]], axis=1)
    timeseries.rename(columns={selected_value: 'value'}, inplace=True)

    aer = AnomaliesAER(timeseries)
    anomalies = pd.DataFrame(aer.detect_anomalies())

    return {'time': time(), "msg_id": body['msg_id'], 'data': str(anomalies.to_csv())}

app = FastStream(broker,  description="Autoencoder")
