from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
from shutil import copyfile
import shutil
import sys
import keras


# MODEL SETTING
img_width, img_height = 150, 150
train_data_dir = './dataset/train/'
validation_data_dir = './dataset/validation/'
nb_train_samples = 4000
nb_validation_samples = 2000
epochs = 2
batch_size = 1

trained_model_file = './weights.best.bigru_3conv_7.hdf5'


def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator


def make_callbacks():
    """
    Callbacks are the set of functions which used every epochs
    return a list of callbacks for building model.
    """
    filepath = "checkpoints/weights.best.bigru_3conv_7.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=0, save_best_only=True, mode='min')
    return [checkpoint]


def build_model():
    """
    return a myterious model
    """

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

    # FC layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


def iterative_train():
    """
    Train model with generators
    """
    train_generator, validation_generator = build_generators()
    model = build_model()
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size, callbacks=make_callbacks())


def predict():

    model = build_model()
    model.load_weights(trained_model_file)
    print(model)
    # model.predict()


def make_testgen():
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return test_datagen.flow_from_directory(
        './testset',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

def main():

    model = build_model()
    model.load_weights(trained_model_file)
    testgen = make_testgen()
    for sample in testgen:

        pr = model.predict(sample[0])
        print(pr, sample[1])
        # print(r)

    # iterative_train()
    # predict()

if __name__ == '__main__':
    main()
