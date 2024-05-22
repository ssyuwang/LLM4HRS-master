
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os, gzip



def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir



def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
def normalization (data, parameters=None):
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
      #min_val[i] = np.nanmin(norm_data[:,i])
      min_val[i] = norm_data[norm_data[:,i].nonzero(),i].min()

      norm_data[:,i] = norm_data[:,i] - norm_data[norm_data[:,i].nonzero(),i].min()
      max_val[i] = norm_data[norm_data[:,i].nonzero(),i].max()
      #max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (norm_data[norm_data[:,i].nonzero(),i].max() + 1e-6)   
      
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

