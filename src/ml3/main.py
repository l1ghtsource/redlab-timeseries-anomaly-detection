from faststream import FastStream
from faststream.kafka import KafkaBroker
from time import time, sleep
from random import random
import clickhouse_connect


broker = KafkaBroker("kafka:9092", max_request_size=16000000)
import pandas as pd

from prophet import Prophet


class ProphetDetector:
    def __init__(self, interval_width=0.999, changepoint_range=0.99, changepoint_prior_scale=0.5):
        self.interval_width = interval_width
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale

    def preprocessing(self, df):
        df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})

        return df

    def fit_predict(self, df):
        df = self.preprocessing(df)

        model = Prophet(daily_seasonality=True, weekly_seasonality=True,
                        seasonality_mode='additive',
                        interval_width=self.interval_width, changepoint_range=self.changepoint_range,
                        changepoint_prior_scale=self.changepoint_prior_scale)

        model = model.fit(df)
        preds = model.predict(df)
        preds['fact'] = df['y'].reset_index(drop=True)

        preds = preds[['ds', 'yhat_lower', 'yhat_upper', 'fact']].copy()

        preds['anomaly'] = 0
        preds.loc[preds['fact'] > preds['yhat_upper'], 'anomaly'] = 1
        preds.loc[preds['fact'] < preds['yhat_lower'], 'anomaly'] = -1

        preds = preds[(preds['anomaly'] == 1) | (preds['anomaly'] == -1)]
        preds = preds.drop(columns=['anomaly', 'yhat_lower', 'yhat_upper'])
        preds = preds.rename(columns={'ds': 'timestamp', 'fact': 'value'})

        return preds



default_client = clickhouse_connect.get_client(host='83.166.235.106', port=8123)
timeseries_all = default_client.query_df('SELECT timestamp as time, web_response, throughput, apdex, error FROM "default"."test2" ORDER BY time ASC')
timeseries_all.rename(columns={'time': 'timestamp'}, inplace=True)

@broker.subscriber("to_ml3")
@broker.publisher("from_ml3")
async def base_handler(body):
    if body['data_source'] == 'default':
       pass
    
    
    selected_value = body['column_name']
    timeseries = pd.concat([timeseries_all['timestamp'], timeseries_all[selected_value]], axis=1)
    timeseries.rename(columns={selected_value: 'value'}, inplace=True)

    detector = ProphetDetector()
    anomalies = detector.fit_predict(timeseries)


    return {'time':time(), "msg_id": body['msg_id'], 'data': str(anomalies.to_csv())}

app = FastStream(broker,  description="Prophet")