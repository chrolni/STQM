#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

            
            
def plot_cs(output_im,im, sensors):
    
    print('sensors',sensors)
    true = im.cpu().numpy() 
    
    np.save('Result/loop_09.npy', output_im)
    
