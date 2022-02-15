import pandas as pd
from dask import dataframe as dd


##### PUT YOUR PATH #####
DATA_PATH = "/gdrive/My Drive/Megafon/"
DATA_PATH = ""
print(DATA_PATH)


def load_file(fname: str):
    _d = pd.read_csv(f'{fname}.csv', low_memory=True)
    print(f"loaded {_d.shape} from {fname}")
    return _d

def load_file_dask(fname: str):
    _d = dd.read_csv(f'{fname}.csv', delimiter='\t')
    print(f"loaded from {fname}")
    return _d

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


if __name__ == "__main__":
    print(f"Start join {DATA_PATH}features and {DATA_PATH}data_test datasets...")
    # comment to continue
    exit()
    ##### Merge profiles and users data from features file (dask) #####
    # train set
    # trn_features = join_profiles_users_from_features_file(dskdf=load_file_dask(f"{DATA_PATH}features"), usrs=load_file(f"{DATA_PATH}data_train"), fname='data_train', validate=True)
    # test set
    tst_features = join_profiles_users_from_features_file(dskdf=load_file_dask(f"{DATA_PATH}features"), usrs=load_file(f"{DATA_PATH}data_test"), fname='data_test', validate=True)

    ##### Merge profiles and users data from pre-saved profiles (in pandas) #####

    # ftrain = load_file('data_train')
    # ptrn = load_file('data_train_id_profiles') # profiles/features df
    # trn_features = join_profiles_users(prfl=ptrn, usrs=ftrain, fname ='data_train', validate=True)
    # trn_features = join_profiles_users(prfl=load_file('data_train_id_profiles'), usrs=load_file(f"{DATA_PATH}data_train"), fname ='data_train', validate=True)

    # ftest = load_file('data_test')
    # ptst = load_file('data_test_id_profiles') # profiles/features df
    # tst_features = join_profiles_users(prfl=ptst, usrs=ftest, fname ='data_test', validate=True)
    # tst_features = join_profiles_users(prfl=load_file('data_test_id_profiles'), usrs=load_file(f"{DATA_PATH}data_test"), fname ='data_test', validate=True)



