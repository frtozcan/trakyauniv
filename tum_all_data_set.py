# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:56:47 2021

@author: FO_KLU
"""

import os
#os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from tensorflow import keras
#import segmentation_models as sm
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
#from model import build_unet
from tensorflow.keras.utils import plot_model
from fognet import fognet
from metrics import dice_coef, iou
from models import Attention_ResUNet, UNet, Attention_UNet,dice_coef_loss, jacard_coef

######################### RESİM BOYUTLARI ######################

H = 256
W = 256

input_shape=(H, W, 3)

###########  MODELİ SEÇ  #####################

"""
### MODEL isimleri: 
                    UNet
                    Attention_UNet
                    Attention_ResUNet
                    fognet
                    LinkNet
"""
MODEL_NAME="fognet"


if MODEL_NAME == 'fognet':
    model = fognet(input_shape)
    
elif MODEL_NAME == "UNet":
    model = UNet(input_shape)
    
elif MODEL_NAME == "Attention_UNet":
    model = Attention_UNet(input_shape)
    
elif MODEL_NAME == "Attention_ResUNet":
    model = Attention_ResUNet(input_shape)
   
elif MODEL_NAME == 'INCEPTION' :
    model = UNet(input_shape)



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split=0.3):
    images = sorted(glob(os.path.join(dataset_path, "E:/data3/train/img", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "E:/data3/train/mask", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

   

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x                                ## (256, 256, 3)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)                    ## (256, 256)
    x = np.expand_dims(x, axis=-1)              ## (256, 256, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Folder for saving data """
yol="D:/sm/spider/MODELLER/" + MODEL_NAME + "/files/"
create_dir(yol)

#create_dir("files")

""" Hyperparameters """
batch_size = 4
lr = 1e-3 ## (0.0001)
num_epoch = 100

model_path = yol + MODEL_NAME+".h5"
csv_path = yol +MODEL_NAME+".csv"

#model_path = "files/model_256_28_2.h5"
#csv_path = "files/data_256_28_2.csv"

""" Dataset : 60/20/20 """
dataset_path = "E:/data3/train/"
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")


train_dataset = tf_dataset(train_x, train_y, batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

train_steps = len(train_x)//batch_size
valid_steps = len(valid_x)//batch_size

if len(train_x) % batch_size != 0:
    train_steps += 1

if len(valid_x) % batch_size != 0:
    valid_steps += 1
    
print (train_steps)
print (valid_steps)


""" Model """
#model = build_unet((H, W, 3))
#model = fognet((H, W, 3))
metrics = ["accuracy", dice_coef, iou, Recall(), Precision()]


model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)


# ADADELTA+binary_crossentropy
#model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'), metrics=metrics)
#ADAGRAD+binary_crossentropy
#model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad'), metrics=metrics)


model.summary()
plot_model(model,to_file= yol + MODEL_NAME + ".png")


callbacks=TensorBoard(
log_dir=yol +'/logs', histogram_freq=0, write_graph=True,
write_images=False,  update_freq='epoch',
profile_batch=2, embeddings_freq=0, embeddings_metadata=None)
"""
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-9, verbose=2),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

"""


from datetime import datetime
start1 = datetime.now()

history=model.fit(
    train_dataset,
    epochs=num_epoch,
    validation_data=valid_dataset,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks
)

stop1 = datetime.now()
# Execution time of the model
execution_time_Unet = stop1 - start1
print("UNet execution time is: ", execution_time_Unet)


############################################## eğitilmiş modeli KAYDETME VE yükleme


MODEL_NAME="fognet"
yol="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"

model_us = tf.keras.models.load_model(yol+ "fognet20210912.hdf5", 
                                   compile=False
                                   )

MODEL_NAME="fognet"
yol="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"

model_list = tf.keras.models.load_model(yol+ "fognet20210919list.hdf5", 
                                   compile=False
                                   )

MODEL_NAME="fognet"
yol="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"

model_chaos= tf.keras.models.load_model(yol+ "fognet20210914chaos.hdf5", 
                                   compile=False
                                   )


##########################################  model geçmişini KAYDETME

import pickle
from datetime import datetime

def save_history(hist, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)



strTemp = yol + MODEL_NAME + datetime.now().strftime("%Y%m%d") +  "_history.h"
save_history(history, strTemp)



############################################## model geçmişini yükleme
    
import pickle
MODEL_NAME="fognet"
history_path="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"
file = open(history_path + "fognet20210912_history.h",'rb')
history_us = pickle.load(file)



MODEL_NAME="fognet"
history_path="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"
file = open(history_path + "fognet20210919_List_history.h",'rb')
history_list = pickle.load(file)



MODEL_NAME="fognet"
history_path="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"
file = open(history_path + "fognet20210914_chaos_history.h",'rb')
history_chaos= pickle.load(file)


############################################## Grafik ÇİZİMLERİ


from matplotlib import pyplot as plt
# plot the training loss at each epoch
loss_Our_dataset = history_us['loss']
loss_LIST_dataset = history_list['loss']
loss_CHAOS_dataset = history_chaos['loss']

epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset loss')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset loss')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset loss')

plt.title('Training loss of Datasets')
#plt.title('Modellerin Eğitim Kayıpları')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

  
#validation loss
loss_Our_dataset = history_us['val_loss']
loss_LIST_dataset = history_list['val_loss']
loss_CHAOS_dataset = history_chaos['val_loss']

epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset loss')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset loss')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset loss')

plt.title('Validation loss of Datasets')
#plt.title('Modellerin Eğitim Kayıpları')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


##############################################################################################

#Training Accuracy of Models
loss_Our_dataset = history_us['accuracy']
loss_LIST_dataset = history_list['accuracy']
loss_CHAOS_dataset = history_chaos['accuracy']

epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset Validation Accuracy')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset Validation Accuracy')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset Validation Accuracy')

plt.title('Training Accuracy of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



#Validation Accuracy of Models
loss_Our_dataset = history_us['accuracy']
loss_LIST_dataset = history_list['accuracy']
loss_CHAOS_dataset = history_chaos['accuracy']

epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset Validation Accuracy')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset Validation Accuracy')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset Validation Accuracy')
plt.title('Validation Accuracy of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


############################################################################################################

#Training Dice of Models
loss_Our_dataset = history_us['dice_coef']
loss_LIST_dataset = history_list['dice_coef']
loss_CHAOS_dataset = history_chaos['dice_coef']
epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset Dice coefficient')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset Dice coefficient')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset Dice coefficient')
plt.title('Training Dice Coefficient of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Dice coefficient')
plt.legend()
plt.show()


#Validation Dice of Models
loss_Our_dataset = history_us['val_dice_coef']
loss_LIST_dataset = history_list['val_dice_coef']
loss_CHAOS_dataset = history_chaos['val_dice_coef']
epochs = range(1, len(loss_Our_dataset) + 1)
plt.plot(epochs, loss_Our_dataset, 'b', label='Our Dataset Dice coefficient')
plt.plot(epochs, loss_LIST_dataset, 'g', label='LIST Dataset Dice coefficient')
plt.plot(epochs, loss_CHAOS_dataset, 'r', label='CHAOS Dataset Dice coefficient')
plt.title('Validation Dice Coefficient of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Dice coefficient')
plt.legend()
plt.show()


############################################################################################

#IOU Dice of Models
iou_fognet = history_us['iou']
iou_UNet = history_list['iou']
iou_Attention_UNet = history_chaos['iou']

epochs = range(1, len(iou_fognet) + 1)
plt.plot(epochs, iou_fognet, 'b', label='Our Dataset iou')
plt.plot(epochs, iou_UNet, 'g', label='LIST Dataset iou')
plt.plot(epochs, iou_Attention_UNet, 'r', label='CHAOS Dataset iou')
plt.title('Training Intersection Over Union of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Intersection Over Union')
plt.legend()
plt.show()


#IOU Dice of Models
iou_fognet = history_us['val_iou']
iou_UNet = history_list['val_iou']
iou_Attention_UNet = history_chaos['val_iou']
epochs = range(1, len(iou_fognet) + 1)
plt.plot(epochs, iou_fognet, 'b', label='Our Dataset val_iou')
plt.plot(epochs, iou_UNet, 'g', label='LIST Dataset val_iou')
plt.plot(epochs, iou_Attention_UNet, 'r', label='CHAOS Dataset val_iou')
plt.title('Validation Intersection Over Union of Datasets')
plt.xlabel('Epochs')
plt.ylabel('Intersection Over Union')
plt.legend()
plt.show()
    