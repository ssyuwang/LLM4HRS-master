
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os, gzip

import pandas as pd


def normalization_grud (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      #norm_data[norm_data[:,i].nonzero(),i].min()
      nonzeros_val = norm_data[norm_data[:,i].nonzero(),i]
      #nonzeros_val = norm_data[~np.isnan(norm_data[:,i]),i]
      '''
      if len(nonzeros_val) > 0 :
         min_val[i] = nonzeros_val.min()
         max_val[i] = nonzeros_val.max()
      '''
      if len(nonzeros_val[0]) > 0 :
         #min_val[i] = norm_data[norm_data[:,i],i].min()
         #max_val[i] = norm_data[norm_data[:,i],i].max()
         min_val[i] = nonzeros_val[0].min()
         max_val[i] = nonzeros_val[0].max()
      
      else:
         min_val[i] = 0
         max_val[i] = 0
        
      norm_data[:,i] = norm_data[:,i] - min_val[i]
     
      #max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    mean_val = norm_parameters['min_val']
    std_val = norm_parameters['max_val']

    #row, dim = tf.shape(norm_data)  # Assuming 'norm_data' is a TensorFlow tensor
    row, dim = norm_data.shape
    renorm_data = tf.identity(norm_data)  # Create a copy of 'norm_data'

    # 创建一个形状为 (dim,) 的 TensorFlow 1.x Tensor，用于存储 std_val 和 1e-6 的加法结果
    scale_factors = tf.constant(std_val + 1e-6, shape=[dim],dtype=tf.float32)

    # 创建一个形状为 (dim,) 的 TensorFlow 1.x Tensor，用于存储 mean_val
    shift_values = tf.constant(mean_val, shape=[dim],dtype=tf.float32)

    # 使用 TensorFlow 1.x 的操作进行批量计算
    renorm_data = tf.multiply(renorm_data, scale_factors)
    renorm_data = tf.add(renorm_data, shift_values)
    '''
    for i in range(dim):
        for j in range(row):
            condition = tf.not_equal(norm_data[j, i], 0)
            # 计算新的值
            new_value = tf.where(condition, 
                                (renorm_data[j, i] * (std_val[i] + 1e-6) + mean_val[i]), 
                                renorm_data[j, i])
            # 更新 Tensor 中的元素
            renorm_data = tf.scatter_update(renorm_data, tf.constant([[j, i]]), new_value)
    '''
    return renorm_data


def renormalization_np (norm_data, norm_parameters):
  
  
  #mean_val = norm_parameters['mean_val']
  #std_val = norm_parameters['std_val']
  mean_val = norm_parameters['min_val']
  std_val = norm_parameters['max_val']

  row, dim = norm_data.shape
  #renorm_data = norm_data.copy()
  renorm_data = np.zeros(shape=(row,dim))
    
  for i in range(dim):
      for j in range(row):
          if norm_data[j,i] !=0:
              renorm_data[j,i] = norm_data[j,i] * (std_val[i] + 1e-6)   
              renorm_data[j,i] = renorm_data[j,i] + mean_val[i]
  
        #renorm_data[:,i] = renorm_data[:,i] * (std_val[i] + 1e-6)   
        #renorm_data[:,i] = renorm_data[:,i] + mean_val[i]
    
  return renorm_data