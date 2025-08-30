import numpy as np
import torch
import random
import pandas as pd

def sea_n_sensors(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    im = np.copy(data[0,]).squeeze()
    print('Picking up sensor locations \n')
    coords = []

    # 10loop
    m=n_sensors # 挑选站点
    total_stations = 319 # 总站点



    lat_lon = pd.read_csv('Data/bay/sensors.csv',index_col=False)
    segment_counts = lat_lon['segment'].value_counts()
    segment_sample_counts = np.round(m * segment_counts).astype(int)
    # 初始化一个空列表来存储选择的列索引
    selected_columns = []
    
    # 遍历每种 segment 类型
    for segment, count in segment_sample_counts.items():
        # 获取该 segment 类型的站点索引
        segment_indices = lat_lon[lat_lon['segment'] == segment].index.tolist()
        # 随机选择 count 个站点索引
        selected_indices = np.random.choice(segment_indices, size=count, replace=False)
        # 将选择的索引添加到列表中
        selected_columns.extend(selected_indices)
    
    # 将选择的列索引转换为 numpy 数组
    selected_columns = selected_columns
    print('selected_columns:',selected_columns)
    coords = [[item, 0] for item in selected_columns]
    coords = np.array(coords)
    print('coords:',coords)


    return coords[:,0], coords[:,1]
                
        
        
    
    

        
        
    
    
    
    
    
    