import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(8,)))
    model.add(layers.Dense(2000, activation='relu'))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(3, activation='tanh'))
    
    optimizer = optimizers.Adam(learning_rate=0.0001)
    
    model.summary()
    model.compile(loss="mse", optimizer=optimizer)
    return model


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


def normalize(row):
    if row['x'] >= 0:
        norm_x = row['x'] / max_x
    else:
        norm_x = row['x'] / -min_x

    if row['y'] >= 0:
        norm_y = row['y'] / max_y
    else:
        norm_y = row['y'] / -min_y

    if row['z'] >= 0:
        norm_z = row['z'] / max_z
    else:
        norm_z = row['z'] / -min_z

    return {'x': norm_x, 'y': norm_y, 'z': norm_z}


def desnormalize(row):
    if row['x'] >= 0:
        desnorm_x = row['x'] * max_x
    else:
        desnorm_x = row['x'] * -min_x

    if row['y'] >= 0:
        desnorm_y = row['y'] * max_y
    else:
        desnorm_y = row['y'] * -min_y
        
    if row['z'] >= 0:
        desnorm_z = row['z'] * max_z
    else:
        desnorm_z = row['z'] * -min_z

    return {'x': desnorm_x, 'y': desnorm_y, 'z': desnorm_z}



def splitData(df: pd.DataFrame, p_train: float, window_size = 3):
    x_start_train = []
    y_start_train = []
    z_start_train = []
    time_start_train = []
    x_end_train = []
    y_end_train = []
    z_end_train = []
    time_end_train = []

    x_target_train = []
    y_target_train = []
    z_target_train = []
    time_target_train = []

    x_start_test = []
    y_start_test = []
    z_start_test = []
    time_start_test = []
    x_end_test = []
    y_end_test = []
    z_end_test = []
    time_end_test = []

    x_real_test = []
    y_real_test = []
    z_real_test = []
    time_real_test = []

    feat_train_data = {}
    feat_test_data = {}
    target_train_data = {}
    real_data = {}
    
    jump = window_size - 1
    for i in range(0, len(df) - jump):
        if (i + jump) <= (len(df) * p_train) - 1:
            for k in range(1, jump):
                x_start_train.append(df.loc[i, "x"])
                y_start_train.append(df.loc[i, "y"])
                z_start_train.append(df.loc[i, "z"])
                time_start_train.append(0)

                x_target_train.append(df.loc[i + k, "x"])
                y_target_train.append(df.loc[i + k, "y"])
                z_target_train.append(df.loc[i + k, "z"])
                time_target_train.append(k)

                x_end_train.append(df.loc[i + jump, "x"])
                y_end_train.append(df.loc[i + jump, "y"])
                z_end_train.append(df.loc[i + jump, "z"])
                time_end_train.append(jump)
                
        elif i >= len(df) - (len(df) * (1-p_train)):
            for k in range(1, jump):
                x_start_test.append(df.loc[i, "x"])
                y_start_test.append(df.loc[i, "y"])
                z_start_test.append(df.loc[i, "z"])
                time_start_test.append(0)

                x_real_test.append(df.loc[i + k, "x"])
                y_real_test.append(df.loc[i + k, "y"])
                z_real_test.append(df.loc[i + k, "z"])
                time_real_test.append(k)

                x_end_test.append(df.loc[i + jump, "x"])
                y_end_test.append(df.loc[i + jump, "y"])
                z_end_test.append(df.loc[i + jump, "z"])
                time_end_test.append(jump)

    feat_train_data['x1'] = x_start_train
    feat_train_data['y1'] = y_start_train
    feat_train_data['z1'] = z_start_train
    
    feat_train_data['x3'] = x_end_train
    feat_train_data['y3'] = y_end_train
    feat_train_data['z3'] = z_end_train
    feat_train_data['t3'] = np.array(time_end_train) / max_t
    feat_train_data['t2'] = np.array(time_target_train) / max_t

    target_train_data['x2'] = x_target_train
    target_train_data['y2'] = y_target_train
    target_train_data['z2'] = z_target_train

    df_feat_train = pd.DataFrame(data=feat_train_data)
    
    df_target_train = pd.DataFrame(data=target_train_data)

    feat_test_data['x1'] = x_start_test
    feat_test_data['y1'] = y_start_test
    feat_test_data['z1'] = z_start_test
    feat_test_data['x3'] = x_end_test
    feat_test_data['y3'] = y_end_test
    feat_test_data['z3'] = z_end_test
    feat_test_data['t3'] = np.array(time_end_test) / max_t
    feat_test_data['t2'] = np.array(time_real_test) / max_t

    real_data['x2'] = x_real_test
    real_data['y2'] = y_real_test
    real_data['z2'] = z_real_test
    
    df_feat_test = pd.DataFrame(data=feat_test_data)
    
    df_real = pd.DataFrame(data=real_data)
    
    return df_feat_train, df_target_train, df_feat_test, df_real


#file = "env1.txt"
file = "real.csv"
env = file.split(".")[0]

if env == "real":
    dados = pd.read_csv(f"./datasets/{file}", sep=',', usecols=['x', 'y'])
    dados['z'] = 1
    size = 15500
    
else:
    dados = pd.read_csv(f"./datasets/{file}", sep=';', usecols=['x', 'y', 'z'])
    #Can be any size until 185000
    size = 50000

min_x = dados.x.min()
max_x = dados.x.max()
min_y = dados.y.min()
max_y = dados.y.max()
min_z = dados.z.min()
max_z = dados.z.max()

size_str = str(int(size/1000)) + "k"

dados_norm = dados.apply(lambda row: normalize(row), axis=1)
dados_norm = pd.DataFrame(dados_norm.tolist())

if env == "real":
    dados_train = dados_norm[:size]
else:
    dados_teste = dados_norm[-100000: ]
    
dados_teste = dados_norm[size: ]
dados_teste.reset_index(inplace=True)

percentage_train = 0.8
times = 10
epochs=200
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')]
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True))

r2_model = []
mse_model = []
rmse_model = []
mae_model = []

path = f"./validation/{env}/rna"
for w in range(3, 33):   
    r2_exec = {'x': [], 'y': [], 'z': []}
    mse_exec = {'x': [], 'y': [], 'z': []}
    rmse_exec = {'x': [], 'y': [], 'z': []}
    mae_exec = {'x': [], 'y': [], 'z': []}
    
    max_t = w - 1
    
    print("Splitting the data...")
    df_feat_train, df_target_train, df_feat_val, df_target_val = splitData(dados_train, percentage_train, window_size=w)
    df_feat_test, df_target_test, _, _ = splitData(dados_teste, 1, window_size=w)
        
    df_target_test.columns = ['x', 'y', 'z']
    df_target_test = df_target_test.apply(lambda row: desnormalize(row), axis=1)
    df_target_test = pd.DataFrame(df_target_test.tolist())
    df_target_test.columns = ['x2', 'y2', 'z2']
    
    df_feat_train.info()
    
    for i in range(times):
        print(w, f"- {i + 1}ª vez")
           
        print("modelling...")
        model = create_model()
        history = model.fit(df_feat_train, df_target_train, validation_data=(df_feat_val, df_target_val), epochs=epochs, callbacks=callbacks)
        model.save(f"./validacao/modelos/{env}/rna/model_d{w-2}")
        
        predicted_points = model.predict(df_feat_test)
        
        df_predito = pd.DataFrame(predicted_points, columns = ['x', 'y', 'z'])
        
        df_predito = df_predito.apply(lambda row: desnormalize(row), axis=1)
        df_predito = pd.DataFrame(df_predito.tolist())
        df_predito.columns = ['x2', 'y2', 'z2']

        model_indicators = validation(df_target_test, df_predito)
        
        (r2_x, r2_y, r2_z), (rmse_x, rmse_y, rmse_z), (mae_x, mae_y, mae_z), (mse_x, mse_y, mse_z) = model_indicators
        print(f"R2: x={r2_x}, y={r2_y}, z={r2_z}")
        print(f"RMSE: x={rmse_x}, y={rmse_y}, z={rmse_z}")
        print(f"MAE: x={mae_x}, y={mae_y}, z={mae_z}")
        print(f"MSE: x={mse_x}, y={mse_y}, z={mse_z}")
        exit()
        
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
        
        pd.DataFrame(r2_exec).to_csv(f"{path}/r2_{size_str}_d{w-2}.csv")
        pd.DataFrame(mse_exec).to_csv(f"{path}/mse_{size_str}_d{w-2}.csv")
        pd.DataFrame(rmse_exec).to_csv(f"{path}/rmse_{size_str}_d{w-2}.csv")
        pd.DataFrame(mae_exec).to_csv(f"{path}/mae_{size_str}_d{w-2}.csv")

        del model
        del df_predito
        
    del df_feat_train
    del df_target_train
    del df_feat_test
    del df_target_test
    del df_feat_val
    del df_target_val 
    print("-----------------------------------------------------------------------------------------")
