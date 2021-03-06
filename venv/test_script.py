from __future__ import absolute_import, division, print_function, unicode_literals
#Setting up imports, using following guide as a reference for the first function
#Courtesty of tensorflow https://www.tensorflow.org/tutorials/generative/style_transfer
import tensorflow as tf; import tensorflow_hub as hub
import IPython.display as display
import matplotlib.pyplot as plt; import  matplotlib as mpl
mpl.rcParams["figure.figsize"] = (12,12)
mpl.rcParams["axes.grid"] = False
import numpy as np; import PIL.Image; import time; import functools
#To get the modules working, use pip install: tensorflow, Image, Mmatplotlib and ipython

#Selects base image, left side is name, right side is file origin
base_image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#base_image_path = tf.keras.utils.get_file('Margaret_Thatcher_portrait.jpg', 'https://upload.wikimedia.org/wikipedia/commons/1/1a/Margaret_Thatcher_portrait.jpg')
# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
base_style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
#base_style_path = tf.keras.utils.get_file('nR3aiaL.jpg','https://i.imgur.com/nR3aiaL.jpg')
#style_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
print("Tensorflow did not explode")

def tensor_to_image(tensor):
    tensor = tensor*255 #pixels
    tensor = np.array(tensor, dtype=np.uint8) #Converts tensor to array using int8 as datatype
    if np.ndim(tensor)>3: #If number of array dimensions is less than 3
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_image(path_to_img):
    max_dim = 512 #pixels
    img = tf.io.read_file(path_to_img) #assing image to vrb
    img = tf.image.decode_image(img, channels=3)#converts bytes of file into tensors
    img = tf.image.convert_image_dtype(img, tf.float32)#converts image into tf.dtype

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)#casts tensor to new shape
    long_dim = max(shape)
    scale = max_dim / long_dim#assigns size

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)#resizes image to fit max size
    img = img[tf.newaxis, :]
    return img

def img_show(image, title=None):
    if len(image.shape) >3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)



def mix_tf_hub():
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)
    plt.subplot(1,1,1)
    img_show(stylized_image)
    plt.show()

def set_up_vg():
    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x , (224,224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
    pr_probabilitiy = vgg(x)
    pr_probabilitiy.shape
    top_prediction = tf.keras.applications.vgg19.decode_predictions(pr_probabilitiy.numpy())[0]
    var = [(class_name, prob) for (number, class_name, prob) in top_prediction]
    print(var)

    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    #print()
    for layer in vgg.layers:
        print(layer.name)
    content_layers = ["block_conv2"]
    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)





content_image = load_image(base_image_path)
style_image = load_image(base_style_path)

plt.subplot(1, 2, 1)
img_show(content_image, "Content Image")
plt.subplot(1, 2,2)
img_show(style_image, "Style Image")
plt.show()
#mix_tf_hub()
set_up_vg()




