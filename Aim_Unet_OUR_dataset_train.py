
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
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
from Aim_Unet import Aim_Unet
from metrics import dice_coef, iou, dice_loss
from models import Attention_ResUNet, UNet, Attention_UNet,dice_coef_loss, jacard_coef
from LinkNet import LinkNet




from tensorflow.keras import backend as K
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


"""
#Ekran kartı hatta vermediğinden emin olmak için

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""



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
MODEL_NAME="Aim_Unet"



if MODEL_NAME == 'fognet':
    model = fognet(input_shape)
    
elif MODEL_NAME == "UNet":
    model = UNet(input_shape)
    
elif MODEL_NAME == "Aim_Unet":
    model = Aim_Unet(input_shape)
    
elif MODEL_NAME == "Attention_UNet":
    model = Attention_UNet(input_shape)
    
elif MODEL_NAME == "Attention_ResUNet":
    model = Attention_ResUNet(input_shape)
    
elif MODEL_NAME == "LinkNet":
    model = LinkNet(input_shape)
    
elif MODEL_NAME == 'INCEPTION' :
    model = UNet(input_shape)





def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split=0.3):
    #images = sorted(glob(os.path.join(dataset_path, "E:/data3/train/img", "*.jpg")))
    #masks = sorted(glob(os.path.join(dataset_path, "E:/data3/train/mask", "*.png")))
    
    images = sorted(glob(os.path.join(dataset_path, "E:/DATASETLER/our_dataset/train/img", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "E:/DATASETLER/our_dataset/train/mask", "*.png")))

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
yol="D:/liver/spider/MODELLER/" + MODEL_NAME + "/files/"
create_dir(yol)

#create_dir("files")

""" Hyperparameters """
batch_size = 3
lr = 1e-3 ## (0.0001)
num_epoch = 100

model_path = yol + MODEL_NAME+"Aim_Unet.h5"
csv_path = yol +MODEL_NAME+"Aim_Unet.csv"

#model_path = "files/model_256_28_2.h5"
#csv_path = "files/data_256_28_2.csv"

""" Dataset : 60/20/20 """
#dataset_path = "D:/data3/train/"

dataset_path = "E:/DATASETLER/our_dataset/train"

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
metrics = ["accuracy", dice_coef, iou, Recall(), Precision(),dice_loss]


model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
#model.compile(optimizer=Adam(lr = 1e-3), loss=binary_focal_loss(gamma=2), metrics=metrics)

# ADADELTA+binary_crossentropy
#model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'), metrics=metrics)
#ADAGRAD+binary_crossentropy
#model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad'), metrics=metrics)
#ADADELTA+binary_focal_loss
#model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'), loss=binary_focal_loss(gamma=2), metrics=metrics)




model.summary()
plot_model(model,to_file= yol + MODEL_NAME + ".png")
plot_model( model,to_file="./Aim_Unet_mimari.png", show_shapes=True,show_layer_names=True)

for layer in model.layers:
	# check for convolutional layer
	if 'conv2d_transpose' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)






callbacks=TensorBoard(
log_dir=yol +'/logs', histogram_freq=0, write_graph=True,
write_images=False,  update_freq='epoch',
profile_batch=3, embeddings_freq=0, embeddings_metadata=None)
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
print("Aim_UNet execution time is: ", execution_time_Unet)





############################################## eğitilmiş modeli KAYDETME VE yükleme

   
from datetime import datetime
model.save(yol+ MODEL_NAME+ datetime.now().strftime("%Y%m%d") + 'Aim_Unet_Our_dataset.hdf5')


##################################MODEL LOAD



model = tf.keras.models.load_model(yol+ "fognet20210919list.hdf5", 
                                   compile=False
                                   )


##########################################  model geçmişini KAYDETME


import pickle
from datetime import datetime

def save_history(hist, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)



strTemp = yol + MODEL_NAME + datetime.now().strftime("%Y%m%d") +  "_Aim_Unet_Our_Dataset_history.h"
save_history(history, strTemp)



############################################## model geçmişini yükleme
 
import pickle   
history_path="./files/"

file = open(history_path + "256_28_2_history.h",'rb')
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
plt.title('Training and validation Dice coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice coefficient')
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
