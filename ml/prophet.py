from prophet import Prophet


class ProphetDetector:
    def __init__(self, interval_width = 0.999, changepoint_range = 0.99, changepoint_prior_scale=0.5):
        self.interval_width = interval_width
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale

    def preprocessing(self, df):
        df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})

        return df

    def fit_predict(self, df):
        df = self.preprocessing(df)

        model = Prophet(daily_seasonality = True, weekly_seasonality = True,
                seasonality_mode = 'additive',
                interval_width = self.interval_width, changepoint_range = self.changepoint_range,
                changepoint_prior_scale = self.changepoint_prior_scale)

        model = model.fit(df)
        preds = model.predict(df)
        preds['fact'] = df['y'].reset_index(drop = True)

        preds = preds[['ds', 'yhat_lower', 'yhat_upper', 'fact']].copy()

        preds['anomaly'] = 0
        preds.loc[preds['fact'] > preds['yhat_upper'], 'anomaly'] = 1
        preds.loc[preds['fact'] < preds['yhat_lower'], 'anomaly'] = -1

        preds = preds[(preds['anomaly'] == 1) | (preds['anomaly'] == -1)]
        preds = preds.drop(columns=['anomaly', 'yhat_lower', 'yhat_upper'])
        preds = preds.rename(columns={'ds': 'timestamp', 'fact': 'value'}).reset_index(drop=True)

        return preds