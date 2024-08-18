# -*- coding: UTF-8 -*-
import pickle
import gzip

# 載入Model
with gzip.open('./model/xgboost-iris.pgz', 'rb') as f:
    xgboostModel = pickle.load(f)

def predict(input):
    pred=xgboostModel.predict(input)[0]
    return pred