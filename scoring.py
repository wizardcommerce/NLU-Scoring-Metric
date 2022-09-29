from __future__ import annotations

import numpy as np
import pandas as pd
import torch


"""
Some docs would be nice for this section in addition to methods.ipynb
"""


class scoring():

    def __init__(self, path_to_csv=None, df=None):
        super().__init__()
        if path_to_csv != None:
            self.df = pd.read_csv(path_to_csv)
        else:
            self.df = df

    def __getitem__(self, col):
        return self.df[col]

    def fj(self, pix_col, fJ_col):

        M = (~self.df[fJ_col].isna().values).astype(float)
        I = self.df[pix_col].values.reshape(-1, 1)

        k = (I * M).sum(axis=0)
        N = (M.sum(axis=0)-1)

        Y = ((k - I) * M) / N
        Y = np.nan_to_num(Y, posinf=0., neginf=0.)
        Y = (M == 0).astype(float) + Y

        return np.prod(Y, axis=-1)

    def data(self, col=None):
        if bool(col):
            return self.df[col]
        else:
            return self.df

    def get_cols(self):
        return self.df.columns.values

    def S(self, pix_col, feature_cols):
        fJ = self.fj(pix_col, feature_cols)
        return self.df[pix_col].values * fJ


class normal():

    def __init__(self, path_to_csv=None, df=None):
        super().__init__()
        if path_to_csv != None:
            self.df = pd.read_csv(path_to_csv)
        else:
            self.df = df
        self.N = None

    def pN(self, x):
        denom = (self.N.log_prob(self.N.loc.view(-1)) *
                 torch.eye(self.N.loc.shape[0])).sum(dim=0)
        return torch.exp(self.N.log_prob(x).T)/torch.exp(denom.unsqueeze(0))

    def paramsN(self, I, M):
        F = (I * M)
        k = F.sum(dim=0)
        N = M.sum(dim=0)

        mu = k / N
        sigma = torch.sqrt(((F - mu) ** 2 * M).sum(dim=0) / N) + 1e-9
        self.N = torch.distributions.Normal(
            loc=mu.view(-1, 1), scale=sigma.view(-1, 1))

    def pc(self, pix_col, fJ_col):

        M = torch.FloatTensor((~self.df[fJ_col].isna().values).astype(float))
        I = torch.FloatTensor(self.df[pix_col].values.reshape(-1, 1))
        self.paramsN(I, M)

        p = (self.pN(I.view(-1)) * M) + (M == 0).float()

        return p.prod(dim=-1)

    def data(self, col=None):
        if bool(col):
            return self.df[col]
        else:
            return self.df

    def get_cols(self):
        return self.df.columns.values

    def S(self, pix_col, feature_cols):
        fJ = self.pc(pix_col, feature_cols)
        return self.df[pix_col].values * fJ.numpy()

    def __getitem__(self, col):
        return self.df[col]

    def __call__(self, pix_col, feature_cols):
        return self.S(pix_col, feature_cols)
