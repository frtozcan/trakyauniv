# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:12:47 2021

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
#from train import load_data, create_dir
from matplotlib import pyplot as plt
from models import Attention_ResUNet, UNet, Attention_UNet,dice_coef_loss, jacard_coef



MODEL_NAME="fognet"

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
yol="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"

#yol="E:/spider/liver/UNET/MODELLER/" + MODEL_NAME + "/files/"

""" Load the model """



with CustomObjectScope({'loss':"loss",'acc':"accuracy", 'dice_coef': dice_coef,'iou': iou,'recall':"recall",'precision':"precision"}):
    model = tf.keras.models.load_model(yol + "fognet20210919list.hdf5")


        
       
#path = 'D:/data3/test/img/'
path = 'E:/DATASETLER/meg_dataset/test/img/'

filename_list = os.listdir(path)

SCORE = []

IoU_values = []
    
for filename in filename_list:#[1:25]:
    
         
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
    y=gt
    gt[gt>0] = 1
        
    input1=np.expand_dims(x, axis=0)
    
        
    decoded_img = model.predict(input1)
    #print("predicted max: ", np.max(decoded_img))
         
    result = decoded_img[0,:,:,0]
    result = result> 0.8
    y_pred= result

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
    name = dosya_adi[0].split("_")[-1]
    #tahmin edilen resmi Kaydediyoruz
    #cv2.imwrite(f"data/predict/{ad}.jpg", predicted)
    
    
    
    cv2.putText(image_result, filename, (10, 50), 1, 1, (0, 0, 255))
    
    sonuc=image_result+image1
    cv2.imshow("image-result", image_result)
   
    
    
    
    """
    
    cv2.imwrite("./MODELLER/"+MODEL_NAME+"/files/result/"+ name +"_mask.png", gt)
    cv2.imwrite("./MODELLER/"+MODEL_NAME+"/files/result/"+ name +"_pred.jpg", predicted)
    cv2.imwrite("./MODELLER/"+MODEL_NAME+"/files/result/"+ name +"_orj.jpg", image1)
    cv2.imwrite("./MODELLER/"+MODEL_NAME+"/files/result/"+ name +"_gather.jpg", sonuc)
    """
   
    
    
    
    # IoU for a single image
    from tensorflow.keras.metrics import MeanIoU
    
    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y, y_pred)
    #print("Mean IoU =", IOU_keras.result().numpy())
    IoU = IOU_keras.result().numpy()
    IoU_values.append(IoU)

    #print(IoU)
    
    """ Flatten the array """
    y = y.flatten()
    y_pred = y_pred.flatten()
    
    """ Calculating metrics values """
    acc_value = accuracy_score(y, y_pred)
    f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
    jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
    recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
    precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
    SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    
    cv2.waitKey(0)
    
    
cv2.destroyAllWindows()



""" mean metrics values """
print("#"*60)

score = [s[1:] for s in SCORE]
score = np.mean(score, axis=0)
print(f"Accuracy: {score[0]:0.5f}")
print(f"F1: {score[1]:0.5f}")
print(f"Jaccard: {score[2]:0.5f}")
print(f"Recall: {score[3]:0.5f}")
print(f"Precision: {score[4]:0.5f}")

df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
df.to_csv("files/score.csv")        

  
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)   
    


############################################## eğitilmiş modeli YÜKLEME
    
from datetime import datetime
model_path = yol

model = tf.keras.models.load_model(model_path + "fognet20210901.hdf5", compile=False)




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



    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  