import numpy as np
import pandas as pd

class scoring():

    def __init__(self, path_to_csv):
        super(scoring, self).__init__()
        self.df = pd.read_csv(path_to_csv)

    def __getitem__(self, col):
        return self.df[col]

    def fj(self, pix_col, fJ_col):

        M = (~self.df[fJ_col].isna().values).astype(float)
        I = self.df[pix_col].values.reshape(-1,1)

        k = (I * M).sum(axis=0)
        N = (M.sum(axis=0)-1)

        Y = ((k - I) * M) / N
        Y = np.nan_to_num(Y, posinf=0.,neginf=0.)
        Y = (M == 0).astype(float) + Y

        return np.prod(Y, axis=-1)

    def data(self,col=None):
        if bool(col):
            return self.df[col]
        else:
            return self.df

    def get_cols(self):
        return self.df.columns.values

    def S(self, pix_col, feature_cols):
        fJ = self.fj(pix_col,feature_cols)
        return self.df[pix_col].values * fJ