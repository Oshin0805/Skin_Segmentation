import numpy as np
import argparse
import tensorflow as tf
from skimage.io import imsave
from PIL import Image
from os.path import join
import glob 

from keras import backend as K
from keras.models import load_model

from model.util import preprocess_input
from model.loss import per_pixel_softmax_cross_entropy_loss, IOU

custom_objects_dict = {
    'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
    'IOU': IOU
}

name_counter = 0

def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
            help="Debug mode: more verbose, test things with less data, etc.",
            action='store_true')
    parser.add_argument("--demo",
            help='Demo the segmentation',
            action='store_true')
    parser.add_argument("--load_path",
            help="optional path argument, if we want to load an existing model")
    args = parser.parse_args()
    return args

# https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
def load_image(path):
    img = Image.open(path)
    oldsize = img.size

    img = img.resize((224, 224), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype='int32')
    
    return data, oldsize

def save_image(array, path, size):
    img = Image.fromarray((array*255).astype(np.uint8), 'L')
    # Reshape it back up to the original size, since we shrunk to (224,224) at first
    img = img.resize(oldsize, Image.ANTIALIAS)
    img.save(path)


if __name__ == "__main__":
    args = parse_arguments_from_command()
    demo = args.demo
    stored_model_path = args.load_path
            
    if stored_model_path is None:
        stored_model_path = input("Load model from rel path: ")

    model = load_model(stored_model_path, custom_objects=custom_objects_dict)
   
    count = 1
    files = glob.glob("/home/oshin/Desktop/code/segmentation-191/demo-images/complex/*.jpg")
    for file in files:
        array, oldsize = load_image(file)
        original= array
        array = array.astype(np.float64)
        array = array[np.newaxis, ...]
        array = preprocess_input(array)

        y = model.predict(array, verbose=1)
        y = np.argmax(y, axis=-1)
        y = np.squeeze(y) # removebatch dim
        y1 = np.zeros((224,224,3))
        for i in range(224):
            for j in range(224):
                y1[i][j][0] = y[i][j]
                y1[i][j][1] = y[i][j]
                y1[i][j][2] = y[i][j]
        z = np.multiply(y1,original)
        # save_image(y, join("demo-images", filename+"-pred.jpg"), size=oldsize)
        # save_image(z, join("demo-images", filename+"-pred1.jpg"), size=oldsize)
        img = Image.fromarray(z.astype(np.uint8))
        img = img.resize(oldsize, Image.ANTIALIAS)
        img.save(join("demo-images/3", str(count)+"-pred.jpg"))
        count += 1
        print(count)


    #########################
    # Processing images for putting on the poster, predicting on all these files.

    files = [
        '000000100238',
        '000000100510',
        '000000100624',
        '000000100723',
        '000000200421',
        '000000200839',
        '000000200961',
        '000000300276',
        '000000300341',
        '000000400044', # 10
        '000000400573',
        '000000400803',
        '000000400815',
        '000000500270',
        '000000500478', # 15
        '000000500565'
    ]
    for filename in files:

        input_dir = "sample-backpack/people-val"
        output_dir = "assets/output_imgs"

        array, oldsize = load_image(join(input_dir, filename + ".jpg"))

        array = array.astype(np.float64)
        array = array[np.newaxis, ...] # add batch dim
        array = preprocess_input(array) # preprocess for DenseNet
        
        y = model.predict(array, verbose=1)
        y = np.argmax(y, axis=-1)
        y = np.squeeze(y) # remove batch dim
        save_image(y, join(output_dir, filename + "-pred.jpg"), size=oldsize)
