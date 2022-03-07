import os
import gc
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import *
import numpy as np
import pandas as pd
from scipy import stats
import pickle
import gzip
from xgboost import XGBRegressor

def getModel(model_choice: int = 0):
    if model_choice == 0:
        return RandomForestRegressor(max_features="sqrt", min_samples_split=6, n_jobs=12)
    if model_choice == 1:
        base = DecisionTreeRegressor(max_depth=18, min_samples_split=14)
        return AdaBoostRegressor(n_estimators=50, base_estimator=base, loss="linear", learning_rate=0.1)
    if model_choice == 2:
        return XGBRegressor(n_estimators=400, max_depth=12, learning_rate=0.1, subsample=0.1, colsample_bytree=1.0, n_jobs=8)
    if model_choice == 3:
        model = HistGradientBoostingRegressor(learning_rate=0.3,
                                    loss='squared_error'
                                    max_bins=100,
                                    max_iter=1000)
        return model


def interpol(df, col_x1, col_y1, col_z1, col_x3, col_y3, col_z3, col_t3, col_target):
    def get_proporcional_interpolation(row):

        x_diff = row[col_x3] - row[col_x1]
        y_diff = row[col_y3] - row[col_y1]
        z_diff = row[col_z3] - row[col_z1]

        perc_total = (row[col_target]) / row[col_t3]

        x = row[col_x1] + x_diff * perc_total
        y = row[col_y1] + y_diff * perc_total
        z = row[col_z1] + z_diff * perc_total
        return {"x2": x, "y2": y, "z2": z}

    list_dicts = list(df.apply(lambda row: get_proporcional_interpolation(dict(row)), axis=1))
    df_interpoled = pd.DataFrame(list_dicts)

    df_interpoled.columns = ["x2", "y2", "z2"]
    return df_interpoled


def validation(df_test: pd.DataFrame, df_predito: pd.DataFrame):
    x_real = df_test['x2']
    x_previsto = df_predito['x2']
    z_previsto = df_predito['z2']
    
    rmse_x = mean_squared_error(x_real, x_previsto) ** (1 / 2)
    _, _, r_value, _, _ = stats.linregress(x_real, x_previsto)
    r2_x = r_value*r_value
    mae_x = mean_absolute_error(x_real, x_previsto)
    mse_x = mean_squared_error(x_real, x_previsto)

    y_real = df_test['y2']
    y_previsto = df_predito['y2']
    rmse_y = mean_squared_error(y_real, y_previsto) ** (1 / 2)
    _, _, r_value, _, _ = stats.linregress(y_real, y_previsto)
    r2_y = r_value*r_value
    mae_y = mean_absolute_error(y_real, y_previsto)
    mse_y = mean_squared_error(y_real, y_previsto)
    
    z_real = df_test['z2']
    z_previsto = df_predito['z2']
    rmse_z = mean_squared_error(z_real, z_previsto) ** (1 / 2)
    _, _, r_value, _, _ = stats.linregress(z_real, z_previsto)
    r2_z = r_value*r_value
    mae_z = mean_absolute_error(z_real, z_previsto)
    mse_z = mean_squared_error(z_real, z_previsto)

    return (r2_x, r2_y, r2_z), (rmse_x, rmse_y, rmse_z), (mae_x, mae_y, mae_z), (mse_x, mse_y, mse_z)


def convert_test_to_32bits(df):
    df['x2'] = df['x2'].astype(np.float32)
    df['y2'] = df['y2'].astype(np.float32)
    df['z2'] = df['z2'].astype(np.float32)
    return df

def convert_train_to_32bits(df):
    df['x1'] = df['x1'].astype(np.float32)
    df['y1'] = df['y1'].astype(np.float32)
    df['z1'] = df['z1'].astype(np.float32)
    df['x3'] = df['x3'].astype(np.float32)
    df['y3'] = df['y3'].astype(np.float32)
    df['z3'] = df['z3'].astype(np.float32)
    df['t3'] = df['t3'].astype(np.int32)
    df['t2'] = df['t2'].astype(np.int32)
    return df


def splitData(df: pd.DataFrame, window_size = 3):
    x_start_train = []
    y_start_train = []
    z_start_train = []
    x_end_train = []
    y_end_train = []
    z_end_train = []
    time_end_train = []

    x_target_train = []
    y_target_train = []
    z_target_train = []
    time_target_train = []

    feat_train_data = {}
    target_train_data = {}
    
    jump = window_size - 1
    for i in range(0, len(df) - jump):
        for k in range(1, jump):
            x_start_train.append(df.loc[i, "x"])
            y_start_train.append(df.loc[i, "y"])
            z_start_train.append(df.loc[i, "z"])

            x_target_train.append(df.loc[i + k, "x"])
            y_target_train.append(df.loc[i + k, "y"])
            z_target_train.append(df.loc[i + k, "z"])
            time_target_train.append(k)

            x_end_train.append(df.loc[i + jump, "x"])
            y_end_train.append(df.loc[i + jump, "y"])
            z_end_train.append(df.loc[i + jump, "z"])
            time_end_train.append(jump)

    feat_train_data['x1'] = x_start_train
    feat_train_data['y1'] = y_start_train
    feat_train_data['z1'] = z_start_train
    
    feat_train_data['x3'] = x_end_train
    feat_train_data['y3'] = y_end_train
    feat_train_data['z3'] = z_end_train
    feat_train_data['t3'] = time_end_train
    feat_train_data['t2'] = time_target_train

    target_train_data['x2'] = x_target_train
    target_train_data['y2'] = y_target_train
    target_train_data['z2'] = z_target_train
    
    df_feat_train = pd.DataFrame(data=feat_train_data)
    df_feat_train = convert_train_to_32bits(df_feat_train)
    
    df_target_train = pd.DataFrame(data=target_train_data)
    df_target_train = convert_test_to_32bits(df_target_train)
    
    return df_feat_train, df_target_train


file = "env1.txt"
#file = "real.csv"
env = file.split(".")[0]

if env == "real":
    dados = pd.read_csv(f"./datasets/{file}", sep=',', usecols=['x', 'y'])
    dados['z'] = 1
    sizes = [15500]
else:
    dados = pd.read_csv(f"./datasets/{file}", sep=',', usecols=['x', 'y', 'z'])
    sizes = [5000, 20000, 35000, 65000, 80000, 95000, 110000, 125000, 140000, 155000, 170000, 185000, 200000]

for size in sizes:
    print(size)
    dados_treino = dados[:size]
    dados_treino.reset_index(inplace=True)
    size_str = str(int(len(dados_treino)/1000)) + "k"
    
    if env != "real":
        dados_teste = dados[-100000:]
    else:
        dados_teste = dados[size:]
    dados_teste.reset_index(inplace=True)

    model_choice = 0
    times = 1

    path_rf = f"./validation/{env}/random_forest"
    path_ada = f"./validation/{env}/adaboost"
    path_il = f"./validation/{env}/interpolation"
    path_xgb = f"./validation/{env}/xgboost"
    path_hist = f"./validation/{env}/histboost"
    
    for w in range(3, 33):        
        r2_exec = {'x': [], 'y': [], 'z': []}
        mse_exec = {'x': [], 'y': [], 'z': []}
        rmse_exec = {'x': [], 'y': [], 'z': []}
        mae_exec = {'x': [], 'y': [], 'z': []}
                
        df_feat_train, df_target_train = splitData(dados_treino, window_size=w)
        df_feat_test, df_real = splitData(dados_teste, window_size=w)
        df_feat_train.info()
            
        for i in range(times):
            print(w, f"- {i + 1} time")

            print("modelando...")
            model = MultiOutputRegressor(getModel(model_choice)).fit(df_feat_train, df_target_train)

            predicted_points = model.predict(df_feat_test)
            df_predito = pd.DataFrame(predicted_points, columns = ['x2','y2', 'z2'])

            model_indicators = validation(df_real, df_predito)
            
            
            (r2_x, r2_y, r2_z), (rmse_x, rmse_y, rmse_z), (mae_x, mae_y, mae_z), (mse_x, mse_y, mse_z) = model_indicators
            print("Stack:")
            print(f"R2: x={r2_x}, y={r2_y}, z={r2_z}")
            print(f"RMSE: x={rmse_x}, y={rmse_y}, z={rmse_z}")
            print(f"MAE: x={mae_x}, y={mae_y}, z={mae_z}")
            print(f"MSE: x={mse_x}, y={mse_y}, z={mse_z}")

            r2_exec['x'].append(r2_x)
            r2_exec['y'].append(r2_y)
            r2_exec['z'].append(r2_z)

            mse_exec['x'].append(mse_x)
            mse_exec['y'].append(mse_y)
            mse_exec['z'].append(mse_z)

            rmse_exec['x'].append(rmse_x)
            rmse_exec['y'].append(rmse_y)
            rmse_exec['z'].append(rmse_z)

            mae_exec['x'].append(mae_x)
            mae_exec['y'].append(mae_y)
            mae_exec['z'].append(mae_z)

            if i == 0:
                f = gzip.open(f'./validacao/modelos/{env}/random_forest/model_d{w-2}.sav', 'wb')
                pickle.dump(model, f)
            del model
            del df_predito
            del predicted_points
            gc.collect()

        continue

        pd.DataFrame(r2_exec).to_csv(f"{path_hist}/r2_{size_str}_d{w-2}.csv")
        pd.DataFrame(mse_exec).to_csv(f"{path_hist}/mse_{size_str}_d{w-2}.csv")
        pd.DataFrame(rmse_exec).to_csv(f"{path_hist}/rmse_{size_str}_d{w-2}.csv")
        pd.DataFrame(mae_exec).to_csv(f"{path_hist}/mae_{size_str}_d{w-2}.csv")

        del df_feat_train
        del df_target_train
        del df_feat_test
        del df_real
        
        # continue
        
        ##### LINEAR INTERPOLATION #####
        df_interpolacao = interpol(df_feat_test, "x1", "y1", "z1", "x3", "y3", "z3", "t3", "t2")
        base_line_indicators = validation(df_real, df_interpolacao)
        (r2_x, r2_y, r2_z), (rmse_x, rmse_y, rmse_z), (mae_x, mae_y, mae_z), (mse_x, mse_y, mse_z) = base_line_indicators
        print("Interpolação:")
        print(f"R2: x={r2_x}, y={r2_y}, z={r2_z}")
        print(f"RMSE: x={rmse_x}, y={rmse_y}, z={rmse_z}")
        print(f"MAE: x={mae_x}, y={mae_y}, z={mae_z}")
        print(f"MSE: x={mse_x}, y={mse_y}, z={mse_z}")
        ##########################################
        
        del df_feat_train
        del df_target_train
        del df_feat_test
        del df_real

        r2_interp = [{'x': r2_x, 'y': r2_y, 'z': r2_z}]
        mse_interp = [{'x': mse_x, 'y': mse_y, 'z': mse_z}]
        rmse_interp = [{'x': rmse_x, 'y': rmse_y, 'z': rmse_z}]
        mae_interp = [{'x': mae_x, 'y': mae_y, 'z': mae_z}]
        
        pd.DataFrame(r2_interp).to_csv(f"{path_il}/r2_{size_str}_d{w-2}.csv")
        pd.DataFrame(mse_interp).to_csv(f"{path_il}/mse_{size_str}_d{w-2}.csv")
        pd.DataFrame(rmse_interp).to_csv(f"{path_il}/rmse_{size_str}_d{w-2}.csv")
        pd.DataFrame(mae_interp).to_csv(f"{path_il}/mae_{size_str}_d{w-2}.csv")
        print("-----------------------------------------------------------------------------------------")