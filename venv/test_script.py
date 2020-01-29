from __future__ import absolute_import, division, print_function, unicode_literals
#Setting up imports, using following guide as a reference for the first function
#Courtesty of tensorflow https://www.tensorflow.org/tutorials/generative/style_transfer
import tensorflow as tf; import IPython.display as display
import matplotlib.pyplot as plt; import  matplotlib as mpl
mpl.rcParams["figure.figsize"] = (12,12)
mpl.rcParams["axes.grid"] = False
import numpy as np; import PIL.Image; import time; import functools

def tensor_to_image(tensor):
    tensor = tensor*255 #pixels
    tensor = np.array(tensor, dtype=np.uint8) #Converts tensor to array using int8 as datatype
    if np.ndim(tensor)>3: #If number of array dimensions is less than 3
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)