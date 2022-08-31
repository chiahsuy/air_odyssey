import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# set feature column names
dep_cat_cols = ['IATA_Code_Marketing_Airline', 'Origin', 'Dest']
dep_cat_cols_route = ['Origin', 'Dest']
dep_cat_cols_carrier = ['IATA_Code_Marketing_Airline']
dep_num_cols = ["lon_dpt", "lat_dpt", "tmpf_dpt", "dwpf_dpt", "relh_dpt", "drct_dpt", "sknt_dpt",
                "vsby_dpt", "gust_dpt", "sky_coverage_dpt", "mist_dpt", "blowing_dpt", "drifting_dpt",
                "dust_dpt",	"widespread_dust_dpt",	"drizzle_dpt",	"funnel_cloud_dpt",	"fog_dpt",
                "smoke_dpt",	"freezing_dpt", "hail_dpt", "small_hail_dpt",	"haze_dpt",	"ice_crystal_dpt",
                "sleet_dpt",	"rain_dpt",	"sand_dpt",	"snow_grains_dpt",	"shower_dpt", "snow_dpt",
                "squal_dpt",	"thunderstorm_dpt"]#,	"vicinity_dpt",	"patch_dpt",	"partial_dpt",	"shallow_dpt"]
dep_time_col = "CRSDepDateTime"
arr_time_col = "CRSArrDateTime"

dep_cancel_col = "Cancelled"
dep_delay_col = "DepDelay"
dep_delay_col_z = "DepDelay_z"


def get_month(ds):
  return pd.to_datetime(ds).dt.month.array.reshape(-1,1)

def get_wday(ds):
  return pd.to_datetime(ds).dt.dayofweek.array.reshape(-1,1)

def get_hour(ds):
  return pd.to_datetime(ds).dt.hour.array.reshape(-1,1)


mt_transformer = Pipeline(
    steps=[("month", FunctionTransformer(get_month)),
           ("cat",  OneHotEncoder())
           ]
)

wd_transformer = Pipeline(
    steps=[("weekday", FunctionTransformer(get_wday)),
           ("cat",  OneHotEncoder())
           ]
)

hr_transformer = Pipeline(
    steps=[("hour", FunctionTransformer(get_hour)),
           ("cat",  OneHotEncoder())
           ]
)

feature_prep = ColumnTransformer(transformers = [('dep_month', mt_transformer , dep_time_col),
                                                 ('dep_wkday', wd_transformer , dep_time_col),
                                                 ('dep_hour', hr_transformer , dep_time_col),
                                                 ('arr_month', mt_transformer , arr_time_col),
                                                 ('arr_wkday', wd_transformer , arr_time_col),
                                                 ('arr_hour', hr_transformer , arr_time_col),
                                                 ('cat', OneHotEncoder(handle_unknown = 'ignore'),dep_cat_cols),
                                                 ('num', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0), dep_num_cols)
                                  ]
                  )

class LinResd_Regressor1(BaseEstimator, RegressorMixin):
    def __init__(self, linpred, respred):
        self.linpred = linpred
        self.respred = respred
    
    def fit(self, X, y):
        X_a = X#.toarray()
        self.linpred.fit(X_a, y) 
        y_hat = self.linpred.predict(X_a)
        residual = y - y_hat 
        #print(y_hat)
        #print(residual)
        #print(y)
        self.respred.fit(X_a, residual)       
        return self
        
    def predict(self, X):
        X_a = X#.toarray()
        y_hat_r = self.linpred.predict(X_a) + self.respred.predict(X_a)
        #print(self.linpred.predict(X_a), self.respred.predict(X_a), y_hat_r)
        return y_hat_r
