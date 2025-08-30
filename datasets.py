import numpy as np
import h5py
import pickle
import pandas as pd 

def time_add(data, week_start, interval=5, weekday_only=False, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1 * (interval // 5)
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
    return day_data, week_data

def bay():
    sst = np.load('Data/bay/pems-bay-20.npy')
    sst = sst[:288*2*10,:]
    print('sst:',sst.shape)
    
    # adj = pd.read_excel('Data/I405/I405.xlsx')
    weather = np.load('Data/bay/weather-20.npy').T
    weather = np.array(weather)
    print('weather:',weather.shape)
    num_frames = 5760

    sea = np.zeros((num_frames, 319, 1, 1))
    for t in range(num_frames):
        sea[t, :, :, 0] = sst[t, :].reshape(319, 1, order='F')
    
    # ------------------------- 时空间上下文 --------------------
    week_start = 6
    interval = 5
    week_day = 7
    day_data, week_data = time_add(sst, week_start, interval=interval, weekday_only=False)
    if len(sst.shape) == 2:
        sst1 = np.expand_dims(sst, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        weather_data = np.expand_dims(weather, axis=-1).astype(int)
        data_context = np.concatenate([sst1, day_data, week_data, weather_data], axis=-1)
        print('data_context:',data_context)
    # ----------------------------- end ------------------------
    return sea, data_context

def fill_nan(data):
    """Fill nan values with the nearest non-nan value"""
    ind = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        data[:, i] = np.interp(ind, ind[~np.isnan(data[:, i])], data[~np.isnan(data[:, i]), i])
    return data

