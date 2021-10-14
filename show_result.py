# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:17:52 2021

@author: FO_KLU
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir
from matplotlib import pyplot as plt
from models import Attention_ResUNet, UNet, Attention_UNet,dice_coef_loss, jacard_coef



MODEL_NAME="Attention_UNet"

H = 256
W = 256
input_shape=(H, W, 3)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (256, 256)
    return ori_x, x


"""
def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

"""

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Folder for saving data """
yol="E:/spider/liver/UNET/" + MODEL_NAME + "/files/"
create_dir(yol)


""" Load the model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef,"jacard_coef":jacard_coef}):
    model = tf.keras.models.load_model("E:/spider/liver/UNET/MODELLER/Attention_UNet/files/Attention_UNet20210831.hdf5")
        
       
path = 'E:/data_histogram/test/img/'
filename_list = os.listdir(path)
    
for filename in filename_list[2:5]:
    
         
    # herhangi bir görüntünün belirlenme süresi için süreyi başlayıyoruz
    #start_time = time.time()
    if not ".jpg" in filename:
      continue
        
    print(filename)
    image1 = cv2.imread(path + filename)
    image1 = cv2.resize(image1, (W, H))
    image_result=image1/255.0
    x = image_result.astype(np.float32)
       
    image_result = image_result.copy()
    
    gt =  cv2.imread(path.replace("img","mask") + filename.replace("jpg", "png"),0)
    gt = cv2.resize(gt, (W, H))
    gt[gt>0] = 1
        
    input1=np.expand_dims(x, axis=0)
    
        
    decoded_img = model.predict(input1)
    #print("predicted max: ", np.max(decoded_img))
         
    result = decoded_img[0,:,:,0]
    result = result> 0.6

    result=np.squeeze(result)
    result=result.astype(np.uint8)
    predicted= result
       
    image_result[gt > 0, 2] = 255
    image_result[result > 0, 1] = 255
    
        
    
    
    gt=gt*255
    predicted= result * 255
    
    
    
    cv2.imshow("img", image1)
    cv2.imshow("gt", gt )
    cv2.imshow("predicted", predicted)
    
    # hangi dosyayı tahmin edeceğimiz dosyanın adını alıyoruz
    dosya_adi = os.path.splitext(filename)
    ad = dosya_adi[0]
    
    #tahmin edilen resmi Kaydediyoruz
    #cv2.imwrite(f"data/predict/{ad}.jpg", predicted)
    
    cv2.putText(image_result, filename, (10, 50), 1, 1, (0, 0, 255))
    cv2.imshow("image-result", image_result)
    
    #print(f"Resmin Tespit süresi { (time.time() - start_time)} saniyedir.") # görüntünün belirlenme süresinin bitişi
    
    
    cv2.waitKey(0)
    
cv2.destroyAllWindows()




############################################## eğitilmiş modeli KAYDETME VE yükleme
    
from datetime import datetime
model_path = yol+ MODEL_NAME

model = tf.keras.models.load_model(model_path + " .hdf5", compile=False)




########################################## LOAD HİSTORY

#Check history plots, one model at a time
import pickle

history_path=yol + MODEL_NAME

file = open(history_path + "history_data2_Attention_ResUNet.h",'rb')
history = pickle.load(file)




############################################## Grafik ÇİZİMLERİ
    
from matplotlib import pyplot as plt
# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
  

 
acc = history.history['accuracy']
 
val_acc = history.history['val_accuracy']
precision = history.history['precision']  

plt.plot(epochs, acc, 'y', label='accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('precision - Training and validation accuracy')
plt.plot(epochs, precision, 'b', label='precision')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()


acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
  
plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()


  

iou = history.history['iou']
val_iou = history.history['val_iou']
   

plt.plot(epochs, iou, 'y', label='Training iou')
plt.plot(epochs, val_iou, 'r', label='Validation iou')

plt.title('Training and validation iou')
plt.xlabel('Epochs')
plt.ylabel('iou')
plt.legend()
plt.show()



    
"""    
    
   
    
    #                        PLOT GRAPHS
    #######################################################################
    #plot the training and validation accuracy and loss at each epoch
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history['jacard_coef']
    #acc = history.history['accuracy']
    val_acc = history['val_jacard_coef']
    #val_acc = history.history['val_accuracy']
    
    plt.plot(epochs, acc, 'y', label='Training Jacard')
    plt.plot(epochs, val_acc, 'r', label='Validation Jacard')
    plt.title('Training and validation Jacard')
    plt.xlabel('Epochs')
    plt.ylabel('Jacard')
    plt.legend()
    plt.show()
    
    
    
    
    #acc = history.history['dice_metric']
    acc = history['accuracy']
    #val_acc = history.history['val_dice_metric']
    val_acc = history['val_accuracy']
    
    plt.plot(epochs, acc, '-bo', label='Training Dice')
    plt.plot(epochs, val_acc, '-ro', label='Validation Dice')
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.show()
    
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  