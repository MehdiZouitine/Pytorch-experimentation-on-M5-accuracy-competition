from DatasetAutoencoderLSTM import DatasetAutoencoderLSTM
import pandas as pd
import numpy as np


class TestDataLoader:
    def test_shape(self):
        df1 = pd.Dataframe(np.random.rand(1000, 200))
        df2 = pd.Dataframe(np.random.rand(1000, 2000))
        df3 = pd.Dataframe(np.random.rand(1000, 4))
        df4 = pd.Dataframe(np.random.rand(1000, 230))
        s1 = 1
        s2 = 100
        s3 = 2
        s4 = 23
        d1 = DatasetAutoencoderLSTM(df=df1, subsamble_coef=s1)
        d2 = DatasetAutoencoderLSTM(df=df2, subsamble_coef=s2)
        d3 = DatasetAutoencoderLSTM(df=df3, subsamble_coef=s3)
        d4 = DatasetAutoencoderLSTM(df=df4, subsamble_coef=s4)
        for i in range(1000):
            assert d1[i].shape[0] == 1 and d1[i].shape[1] == df1.shape[1] // s1
            assert d2[i].shape[0] == 1 and d2[i].shape[1] == df2.shape[1] // s2
            assert d3[i].shape[0] == 1 and d3[i].shape[1] == df3.shape[1] // s3
            assert d4[i].shape[0] == 1 and d4[i].shape[1] == df4.shape[1] // s4
