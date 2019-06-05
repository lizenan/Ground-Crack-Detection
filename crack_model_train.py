from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
import cv2
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN, TensorBoard, EarlyStopping
from keras.optimizers import Adam


LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
BATCH_SIZE = 64
NUM_GPUS = 1
OPTIMIZER='nadam' #rmsprop
BASE_OUTPUT_DIR = "trained_models_official_wo_bn"

def get_input(path,file_name,normalized = True):
    
    full_path = os.path.join(path,file_name)
    img = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
    img = img.reshape(150,150,-1)
    img = img.astype(np.float32)
    if normalized:
        normalizedImg = np.zeros_like(img)
        normalizedImg = cv2.normalize(img,  normalizedImg, -1, 1, cv2.NORM_MINMAX)
    return normalizedImg

def get_output(path,file_name,normalized = True):
    
    full_path = os.path.join(path,file_name)
    label = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(144,144))
    label = label.reshape(-1)
    if normalized:
        normalizedImg = np.zeros_like(label)
        label = cv2.normalize(label,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
        label = label.reshape(-1)
    return label

def image_generator(img_file,label_file, batch_size = 64):
    while True:
        img_files =  sorted(os.listdir(img_file))
        batch_paths = np.random.choice(a = img_files, 
                                         size = batch_size)
        batch_input = []
        batch_output = [] 
        #print(batch_paths)
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            X = get_input(img_file,input_path,normalized=True )
            y = get_output(label_file,input_path, normalized=True)
        batch_input += [ X ]
        batch_output += [ y ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        yield batch_x, batch_y 

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)
    
def custom_loss(y_true,y_pred):
    return tf.losses.sigmoid_cross_entropy(y_true, logits=y_pred)
weight_decay = 0.
classes = 1
input_shape = (150,150,1)
if __name__ == '__main__':
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid',name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="valid", name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block4_pool')(x)
    
    
    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal',name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2048, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    
    x = BilinearUpSampling2D(size=(16, 16))(x)
    
    x = Conv2D(classes, (1, 1), activation='linear',
                   padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(weight_decay),
                   use_bias=False)(x)
    
    
    row, col, channel = input_shape
    
    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    x = Reshape((-1,))(x)
    #x = Flatten()(x)
    model = Model(img_input, x)
    model.summary()
    train_generator = image_generator("train_aug","train_aug_GT",BATCH_SIZE)
    
    callbacks_list = [
            ModelCheckpoint(
                filepath=os.path.join(BASE_OUTPUT_DIR,
                                      "weights_epoch={epoch:02d}-val_loss={val_loss:.4f}.hdf5"),
                save_best_only=True,
                monitor='val_loss'
            ),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=50,
                verbose=1,
                mode='auto'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=2
            ),
            TerminateOnNaN(),
        ]
    
    optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer,
                    loss=binary_crossentropy_with_logits)
    
    model.fit_generator(train_generator,
                              steps_per_epoch=int((31960//BATCH_SIZE)*0.8),#1014545//options.batch_size,#int((1014545//options.batch_size)*0.8), #1014545//options.batch_size,#100//options.batch_size
                              epochs=NUM_EPOCHS,
                              validation_data=train_generator,
                              validation_steps=int((31960//BATCH_SIZE)*0.2),#98975//options.batch_size,#int((1014545//options.batch_size)*0.2),#98975//options.batch_size,#10//options.batch_size
                              callbacks=callbacks_list,
                              verbose=1)
