import numpy as np
import pickle
import pandas as pd
from dask import dataframe as dd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier

##### PUT YOUR VALUES #####
DATA_PATH = "/gdrive/My Drive/Megafon/"
DATA_PATH = "D:\\GeekbrainAI\\Megafon\\"
TEST_FILENAME = 'data_test'  # if template test file passed (with 4 columns), will do the merge with the features
# TEST_FILENAME = 'data_test_buy_time_id_True_backward'
MODEl_FILENAME = 'model_megafon'
print(DATA_PATH, TEST_FILENAME, MODEl_FILENAME)


class DateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        rs_col = []
        for c in self.columns:
            dtc = f"dt{c}"
            X_[dtc] = X_[c].astype('datetime64[s]')
            X_[f"{c}_weekday"] = X_[dtc].dt.weekday
            X_[f"{c}_month"] = X_[dtc].dt.month
            X_[f"{c}_week"] = X_[dtc].dt.isocalendar().week
            X_[f"{c}_weekday"] = X_[f"{c}_weekday"].astype(np.int8)
            X_[f"{c}_month"] = X_[f"{c}_month"].astype(np.int8)
            X_[f"{c}_week"] = X_[f"{c}_week"].astype(np.int8)
            rs_col = rs_col + [f"{c}_weekday", f"{c}_month", f"{c}_week"]
        return X_[rs_col]

def load_file(fname: str):
    _d = pd.read_csv(f'{fname}.csv', low_memory=True)
    print(f"loaded {_d.shape} from {fname}")
    return _d


def load_file_dask(fname: str):
    _d = dd.read_csv(f'{fname}.csv', delimiter='\t')
    print(f"loaded from {fname}")
    return _d


### MERGING staff ###
def join_profiles_users_from_features_file(dskdf: dd.DataFrame, usrs: pd.DataFrame, fname: str = None, validate=False):
    # extract features for relevant users only
    _profiles = dskdf.loc[dskdf['id'].isin(usrs['id'])].compute()
    # save to file
    if fname is not None:
        pfn = f"{fname}_id_profiles.csv"
        _profiles.to_csv(pfn)
        print(f"Saved {pfn} {_profiles.shape}")
    fdf = join_profiles_users(prfl=_profiles, usrs=usrs, fname=fname, validate=validate)
    return fdf


def join_profiles_users(prfl: pd.DataFrame, usrs: pd.DataFrame, fname: str = None, validate=False):
    _on = 'buy_time'
    _by = 'id'
    _allow_exact_matches = True
    _direction = 'backward'

    print(
        f"got features/profiles: {prfl.shape}, data: {usrs.shape}; join with: on={_on} by={_by} allow_exact_matches={_allow_exact_matches} direction={_direction}")

    prfl = prfl.sort_values(by=_on)
    usrs = usrs.sort_values(by=_on)
    #  copy
    prfl['profile_time'] = prfl['buy_time']
    prfl['profile_id'] = prfl['id'].astype('int')

    fdf = pd.merge_asof(left=usrs, right=prfl, on=_on, by=_by, allow_exact_matches=_allow_exact_matches,
                        direction=_direction)

    xx_ = fdf.loc[~fdf['profile_time'].isna()]
    res = xx_.copy()

    if validate:
        res['dbuy_time'] = res['buy_time'].astype('datetime64[s]')
        res['dprofile_time'] = res['profile_time'].astype('datetime64[s]')

        infoset = ['id', 'profile_id', 'vas_id', 'dprofile_time', 'dbuy_time', '1']
        print(res.shape)
        print(res[infoset].sample(3))

        eq_id = res[infoset][res['id'] != res['profile_id']]
        print(f"Errors: wrong ids {eq_id.shape}")

        err = res[infoset][res['profile_time'] > res['buy_time']]
        print(f"Errors: profile after offer {err.shape}")

        eq = res[infoset][res['profile_time'] == res['buy_time']]
        print(f"OK: equal times {eq.shape}")

        prof_before_offr = res[infoset][res['profile_time'] < res['buy_time']]
        print(f"OK: profile before offer {prof_before_offr.shape}")

    if fname is not None:
        n = f"{fname}_{_on}_{_by}_{_allow_exact_matches}_{_direction}.csv"
        res.to_csv(n)
        print(f"Saved {n} {res.shape}")
    return res


##### if template test file passed (with 4 columns), will do the merge with the features, generate answers file and save it #####
##### full path to model and test file #####
def generate_answers_test(test_fname, model_fname):
    df_test = load_file(f"{test_fname}")
    if df_test.shape[1] < 5:
        df_test = join_profiles_users_from_features_file(dskdf=load_file_dask(f"{DATA_PATH}features"),
                                                         usrs=load_file(f"{test_fname}"),
                                                         fname='data_test',
                                                         # if passed save generated file with this name
                                                         validate=True)
    print(df_test.shape)
    print(df_test.info())

    _model = pickle.load(open(f"{model_fname}", 'rb'))
    _predicts = _model.predict_proba(df_test)[:, 1]
    print(_predicts.shape, _predicts)

    df_test['target'] = _predicts
    print(df_test.head(2))

    answers_df = df_test[['id', 'vas_id', 'buy_time', 'target']]
    print(answers_df.head(2))

    ans_file = f'{DATA_PATH}answers_test.csv'
    answers_df.to_csv(ans_file)
    print(f"Saved {ans_file} {answers_df.shape}")
    return


if __name__ == "__main__":
    generate_answers_test(f"{DATA_PATH}{TEST_FILENAME}", f"{DATA_PATH}{MODEl_FILENAME}")
