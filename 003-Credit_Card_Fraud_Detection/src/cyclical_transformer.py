# cyclical_transformer.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='Time'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        time_mod = X[self.time_col].values % 86400
        X['Time_sin'] = np.sin(2 * np.pi * time_mod / 86400)
        X['Time_cos'] = np.cos(2 * np.pi * time_mod / 86400)
        return X.drop(columns=[self.time_col])

    def get_feature_names_out(self):
        return ['Time_sin', 'Time_cos']

    def get_params(self, deep=True):
        return {'time_col': self.time_col}
    