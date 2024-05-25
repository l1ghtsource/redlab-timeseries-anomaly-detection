from sklearn.ensemble import IsolationForest

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
