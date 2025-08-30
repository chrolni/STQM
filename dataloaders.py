import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
import torch.nn as nn
import pandas as pd 
import random
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from datasets import bay

from sensor_loc import (sea_n_sensors)



import datetime
from positional import PositionalEncoder
from positional import cal_lape
from positional import calculate_normalized_laplacian
from positional import get_normalized_adj

from torch.utils.data import DataLoader,Dataset






def load_data(dataset_name, num_sensors, seed=42):
    
    
        
    if dataset_name == 'bay':
        data, data_context = bay()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
       
    else:
        #raise NameError('Unknown dataset')
        print(f'The dataset_name {dataset_name} was not provided\n')
        print('************WARNING************')
        print('*******************************\n')
        print('Creating a dummy dataset\n')
        print('************WARNING************')
        print('*******************************\n')
        data = np.random.rand(1000,150,75,1)
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
        
    print(f'Data size {data.shape}\n')
    print('x_sens',x_sens,'y_sens',y_sens)
    return torch.as_tensor( data, dtype=torch.float ), x_sens, y_sens, torch.as_tensor( data_context, dtype=torch.float )
    
    
    
def senseiver_dataloader(data_config, num_workers=0):
    return DataLoader( senseiver_loader(data_config), batch_size=None, 
                       pin_memory=True, 
                       shuffle = True,
                       num_workers=4
                     )
    

class senseiver_loader(Dataset):
    
    def __init__(self,  data_config):
    
        data_name   = data_config['data_name']
        num_sensors = data_config['num_sensors']
        seed        = data_config['seed']
        
        self.data_name = data_name
        self.data, x_sens, y_sens, self.data_context = load_data(data_name, num_sensors, seed)
        
        total_frames, *image_size, im_ch = self.data.shape
        
        data_config['total_frames'] = total_frames
        data_config['image_size']   = image_size
        data_config['im_ch']        = im_ch
        
        self.training_frames = data_config['training_frames']
        self.batch_frames    = data_config['batch_frames'] 
        self.batch_pixels    = data_config['batch_pixels'] 
        self.temp_dim_tid    = data_config['temp_dim_tid']
        self.temp_dim_diw    = data_config['temp_dim_diw']
        self.temp_dim_wea    = data_config['temp_dim_wea']
        num_batches = int(self.data.shape[1:].numel()*self.training_frames/(
                                            self.batch_frames*self.batch_pixels))
        
        assert num_batches>0
        
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches
        
        if data_config['consecutive_train']:
            self.train_ind = torch.arange(0,self.training_frames)
        else:
            if seed:
                torch.manual_seed(seed)
            self.train_ind = torch.randperm(self.data.shape[0])[:self.training_frames]

        if self.batch_frames > self.training_frames:
            print('Warning: batch_frames bigger than num training samples')
            self.batch_frames = self.training_frames
            
        # sensor coordinates
        sensors = np.zeros(self.data.shape[1:-1])
        
        if len(sensors.shape) == 2:
            sensors[x_sens,y_sens] = 1
        elif len(sensors.shape) == 3: # 3D images
            sensors[x_sens,y_sens[0],y_sens[1]] = 1
        self.sensors,*_ = np.where(sensors.flatten()==1)
        
        print('data_context:',self.data_context.shape)
        print(self.data_context[:,:,3].shape)
        
        print(self.data_context[:,:,3].max())
        print(self.data_context[:,:,3].min())
        # ------------------------ 构建全体的天/周编码  +  天气编码---------------------------
        tem_emb = []
        time_in_day_emb = nn.Embedding(288+1, self.temp_dim_tid)
        time_in_day_emb_real = time_in_day_emb(self.data_context[:,:,1].long())
        tem_emb.append(time_in_day_emb_real.unsqueeze(1))
        
        day_in_week_emb = nn.Embedding(7+1, self.temp_dim_diw)
        day_in_week_emb_real = day_in_week_emb(self.data_context[:,:,2].long())
        tem_emb.append(day_in_week_emb_real.unsqueeze(1))

        time_weather_emb = nn.Embedding(8+1, self.temp_dim_wea)
        time_weather_emb_real = time_weather_emb(self.data_context[:,:,3].long())
        tem_emb.append(time_weather_emb_real.unsqueeze(1))
        # ------------------------------- end ----------------------------------
        
        # -------------------- 拼接不同特征 ------------------------------------
        self.hidden = torch.cat (tem_emb, dim=-1).transpose(1, 3).detach() #(frame,futrue,loop,values) (2304,96,170,1)
        print('hidden',self.hidden.shape)        
        # -------------------- end -------------------------------------------

        # sine-cosine positional encodings
        self.pos_encodings = PositionalEncoder(self.data.shape[1:],data_config['space_bands'])
        
        self.indexed_sensors  = self.data.flatten(start_dim=1, end_dim=-2)[:,self.sensors,].detach() 
        self.sensor_positions = self.pos_encodings[self.sensors,]
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
                                                    self.batch_frames, axis=0).detach() 
        # get non-zero pixels
        self.pix_avail = self.data.flatten(start_dim=1, end_dim=-2)[0,:,0].nonzero()[:,0]
        
        # if seed:
        #     torch.manual_seed(datetime.datetime.now().microsecond) # reset seed
            
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):

        frames = self.train_ind[ torch.randperm( self.training_frames) ][:self.batch_frames]
        
        pixels = self.pix_avail[ torch.randperm(*self.pix_avail.shape) ][:self.batch_pixels]
        
        # ----------------------------- 按传感器 按时间帧 选取数据 ------------------------------------
        hidden_time = self.hidden[frames,]
        hidden_loop  = hidden_time.flatten(start_dim=2, end_dim=-1)[:,:,self.sensors].transpose(1, 2)
        # -----------------------------         end        ----------------------------------------
        
        sensor_values = self.indexed_sensors[frames,]
        sensor_values = torch.cat([sensor_values,self.sensor_positions], axis=-1) # (frame, loop, value/future) (64,8,1) (64,8,128) MLP的输入
        sensor_values = torch.cat([sensor_values,hidden_loop], axis=-1)
        
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0) # (frame,pix,future) (64,64,128)
        
        # --------------------- 查询融入更多信息 ---------------------------------------------
        hidden_coords = self.hidden[frames,]
        print('hidden_coords',hidden_coords.shape)
        hidden_coords = hidden_coords.flatten(start_dim=2, end_dim=-1)[:,:,pixels].transpose(1, 2) # (frame, pix, future) (64,64, 96)
        coords = torch.cat([coords,hidden_coords], axis=-1)
        # ---------------------- end -------------------------------------------------------
        
        field_values = self.data.flatten(start_dim=1, end_dim=-2)[frames,][:,pixels,]
        
        
        return sensor_values, coords, field_values
        
     
    
