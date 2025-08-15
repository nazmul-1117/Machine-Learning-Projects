from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        else:
            raise ValueError("ColumnDropper requires a DataFrame as input.")
        return self


    def transform(self, X):
        import pandas as pd

        # If X is already a DataFrame, this won't hurt
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.drop(columns=self.columns_to_drop, errors='ignore')

