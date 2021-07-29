
import tensorflow as tf
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import efficientnet.tfkeras as efn 
import numpy as np
from keras_video import VideoFrameGenerator,SlidingFrameGenerator
import tensorflow_addons as tfa
from keras_self_attention import SeqSelfAttention
from SpatialAttention import spatial_attention
from LearningRateScheduler import LossLearningRateScheduler

tf.test.is_gpu_available()

dataset = 'UCF101' #'UCF101' or 'HMDB51' dataset folder name
with open(dataset+'/classInd.txt') as f:
# with open('classInd.txt') as f:
    classes = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
classes = [x.strip() for x in classes]
classes.sort()
len(classes)

epoch= 200
img_height, img_width = 299,299
SIZE = (img_height, img_width)
CHANNELS = 3
NBFRAME = 5
sliding_time = 4
Model_input_size = (NBFRAME, img_height, img_width, CHANNELS)
BS =4
seq_len = NBFRAME
stride = 1
lr_init = 1e-4

# Create video frame generator
def frame_generator(video_path,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time):
    data_aug = ImageDataGenerator()
    
    training_data = SlidingFrameGenerator(
        sequence_time=sliding_time,
        classes = classes, 
        glob_pattern = video_path,
        nb_frames = NBFRAME,
        shuffle = True,
        batch_size=BS,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=False)
    return training_data

def fusion_attention_lstm(image_input_shape,n_class,height,width):
    y = Input(shape=(n_class,))
    input_image = Input(shape=image_input_shape)
    eff_model=efn.EfficientNetB3(input_shape=(height, width, 3),
                                 include_top=False,
                                 weights='noisy-student')
    model_backbone = Model(eff_model.input,eff_model.get_layer('block7a_project_bn').output)
    timeDistributed_layer = tf.keras.layers.TimeDistributed(model_backbone)(input_image)
    print("TimeDistributed", timeDistributed_layer.shape)
    
    '''Temporal'''
    t = tf.keras.layers.TimeDistributed(GlobalAveragePooling2D())(timeDistributed_layer)
    t = LSTM(256, return_sequences=True, input_shape=(t.shape[1],t.shape[2]), name="lstm_layer_in")(t)
    t = SeqSelfAttention(attention_activation='sigmoid')(t)
    avg_pool = GlobalAveragePooling1D()(t)
    max_pool = GlobalMaxPooling1D()(t)
    t = concatenate([avg_pool, max_pool])
    
    t = Dropout(0.3)(t)
    print("Temporal: ", t.shape)
    
    '''Spatial'''
    s = tf.math.reduce_mean(timeDistributed_layer, axis=1)   
    s = SeparableConv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(s)
    s = spatial_attention(s)
    s = SeparableConv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(s)
    s = spatial_attention(s)
    s = BatchNormalization()(s)
    a = GlobalAveragePooling2D()(s)
    c = Dropout(0.3)(a)
    print("Spatial: ", s.shape)
    
    
    '''Fusion'''
    f = tf.keras.layers.Concatenate()([c, t])
    f = Dropout(0.3)(f)
    print("Fusion: ", f.shape)
    
    return f,y,input_image

def fc_action(x,n_class,y):
    x = Dense(1024, name="fusion_dense1")(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax',name="action_output")(x)
    return x

def create_model_fusion(image_input_shape,n_class,height,width,lr_init):
    model,y,input_image = fusion_attention_lstm(image_input_shape,n_class,height,width)
    softmax_action = fc_action(model,n_class,y)
    model = tf.keras.models.Model(inputs=input_image, outputs=softmax_action)
    opt = tfa.optimizers.LazyAdam(lr=lr_init)
#     model.load_weights("ucf_model/UCF_MTDNN_2.h5")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model

def lr_step_scheduler(epoch, lr =lr_init):
    if epoch <= 5:
        return 0.000
    if epoch > 5 and epoch <= 10:
        return 0.00005
    if epoch > 10 and epoch <= 20:
        return 0.000005
    if epoch > 20:
        return 0.00001

def run_model_generator(Model_input_size,img_height,img_width,data_train,data_test,epoch,n_split,lr_init):

    history_list = []
    n_class = len(classes)
    model = create_model_fusion(Model_input_size,n_class,img_height, img_width,lr_init)
    model.summary()

    model_path = dataset+"_model_"+str(n_split)+".h5"
    
#     callback_step = tf.keras.callbacks.LearningRateScheduler(lr_step_scheduler) 
    callback_adapt = LossLearningRateScheduler(base_lr=lr_init, lookback_epochs=3)

    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    stop = EarlyStopping(monitor='val_loss', patience = 10,
                          verbose=0, mode='auto', baseline=None, 
                          restore_best_weights=False)
    callbacks = [checkpoint, stop,callback_adapt]
    steps_per_epoch= (9537 * 0.7) // BS
    eval_per_epoch= 100
    history = model.fit_generator(data_train,
                                  epochs=epoch, 
                                  shuffle=True, 
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data = data_test, 
                                  validation_steps=eval_per_epoch,
                                  callbacks=callbacks)

    history_list.append(np.max(model.history.history['val_accuracy']))
    return history_list
    

def main():
        
    print("***Load split 1***")
    train_files =dataset+'/train1/{classname}/*.avi'
    test_files =dataset+'/test1/{classname}/*.avi'
    train_data = frame_generator(train_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    test_data = frame_generator(test_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    split1_acc = run_model_generator(Model_input_size,img_height,img_width,train_data,test_data,epoch,1,lr_init)
    print("Split 1 Accuracy : ",split1_acc)

    print("***Load split 2***")
    train_files =dataset+'/train2/{classname}/*.avi'
    test_files =dataset+'/test2/{classname}/*.avi'
    train_data = frame_generator(train_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    test_data = frame_generator(test_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    split2_acc = run_model_generator(Model_input_size,img_height,img_width,train_data,test_data,epoch,2,lr_init)
    print("Split 2 Accuracy : ",split2_acc)
    
    print("***Load split 3***")
    train_files =dataset+'/train3/{classname}/*.avi'
    test_files =dataset+'/test3/{classname}/*.avi'
    train_data = frame_generator(train_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    test_data = frame_generator(test_files,classes,NBFRAME,BS,CHANNELS,SIZE,sliding_time)
    split3_acc = run_model_generator(Model_input_size,img_height,img_width,train_data,test_data,epoch,3,lr_init)
    print("Split 3 Accuracy : ",split3_acc)
    return split1_acc,split2_acc,split3_acc

if __name__=='__main__':
    acc1,acc2,acc3 = main()
    print("Split 1 Accuracy : ",np.max(acc1))
    print("Split 2 Accuracy : ",np.max(acc2))
    print("Split 3 Accuracy : ",np.max(acc3))
    print("3 Split Accuracy (Mean): ", (np.max(acc1)+np.max(acc2)+np.max(acc3))/3)