from synthpop import Synthpop
import pandas as pd
import pyreadr

def synBar():
    df = pd.read_csv("bar_pass_prediction.csv")
    print(df.dtypes)
    dtype_map = {}

    for k in df.dtypes.keys():
        match df.dtypes[k]:
            case 'float64':
                dtype_map[k] = 'float'
            case 'category':
                dtype_map[k] = 'category'
                df = df.astype({k : "category"})
            case _:
                dtype_map[k]= 'category'
                df = df.astype({k : "category"})

    print(df.dtypes)
    spop = Synthpop()

    spop.fit(df,dtype_map)

    synth_df = spop.generate(len(df))

    print("synthetische data:")
    print(synth_df.head())

    print("aantal NaNs:")
    print(synth_df.isna().sum())

def synSD2011():
    df0 = pyreadr.read_r("SD2011.rda")['SD2011']
    #pd.read_csv("bar_pass_prediction.csv")
    #print(df0.dtypes)
    df = df0[['age', 'unempdur', 'income', 'sex']]
    #print(df.isna().sum())
    #df.to_excel("inputData.xlsx")
    dtype_map ={
        # "age":"float",
        # "unempdur":"float",
        # "income":"float",
        # "sex":"category"
    }

    for k in df.dtypes.keys():
        match df.dtypes[k]:
            case 'float64':
                dtype_map[k] = 'float'
            case 'category':
                dtype_map[k] = 'category'
                df = df.astype({k : "category"})
            case _:
                dtype_map[k]= 'category'
                df = df.astype({k : "category"})

    print(dtype_map)
    #{'sex': 'float', 'race1': 'category', 'ugpa': 'float', 'bar': 'category'}
    # for (k,v) in dtype_map.items():
    #     if v == 'category':
    #         df[k] = df[k].astype('category')


    r = df.dtypes.keys()
    spop = Synthpop(visit_sequence=['age', 'unempdur', 'income_NaN','income', 'sex'])
    spop.fit(df,dtype_map)

    synth_df = spop.generate(len(df))

    print(synth_df.head())


synBar()