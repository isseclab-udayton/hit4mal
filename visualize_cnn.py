from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
from shutil import copyfile
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_width, img_height = 150, 150

class CONFIG(object):
    """Pre process config"""
    class PRE(object):
        scale = 1./255
        shear_range = 0.2
        zoom_range = 0.2
        horizontal_flip = True

    class MODEL(object):
        """Model config"""
        nb_train_samples = 4000
        nb_validation_samples = 2000
        epochs = 2
        batch_size = 16





def build_dataset_generator():
    train_datagen = ImageDataGenerator(
        rescale=CONFIG.PRE.scale,
        shear_range=CONFIG.PRE.shear_range,
        zoom_range=CONFIG.PRE.zoom_range,
        horizontal_flip=CONFIG.PRE.horizontal_flip)

    validation_datagen = ImageDataGenerator(rescale=CONFIG.PRE.scale)

    train_generator = train_datagen.flow_from_directory(
        'train/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator


def create_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def make_callbacks():
    filepath = "checkpoints/weights.best.bigru_3conv_7.hdf5"
    check_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                               verbose=0, save_best_only=True, mode='min')

    return [check_cb]

def iterative_train():
    model = create_model()
    train_generator, validation_generator = build_dataset_generator()
    callbacks = make_callbacks()

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size, callbacks = callbacks)


def load_model(path):
    model = create_model()
    model.load_weights(path)

    print(model.predict)


if __name__ == '__main__':
    load_model('weights.best.bigru_3conv_7.hdf5')
