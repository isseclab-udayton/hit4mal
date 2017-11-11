import numpy
numpy.random.seed(42)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
from keras.callbacks import ModelCheckpoint

def make_callbacks():
    tb = keras.callbacks.TensorBoard(
        log_dir='./Graph8', histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath='./checkpoints/weights.hdf5',monitor='val_acc', verbose=1, save_best_only=True)
    # es = keras.callbacks.EarlyStopping(
    #     monitor='val_acc', patience=3, mode='max')
    return [tb, checkpointer]
# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'dataset8/train'
validation_data_dir = 'dataset8/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 200
batch_size = 16

def build_model():

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



    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def train_val_generator():
    # this is the augmentation configuration we will use for training
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

def iterative_train():
    train_generator, validation_generator = train_val_generator()
    model = build_model()


    model.fit_generator(
        train_generator,
        callbacks = make_callbacks(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')


def visualize():
    # load model
    model = build_model()
    model = model.loads_weight('./checkpoints/weights_bestmodel.hdf5')
    print('Model loaded.')
    model.summary()

    # place holder for model input
    input_img = model.input
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


# def compare(benign_sample, malicious_sample):




def main():
    iterative_train()

    benign_sample = "./output_dataset/benign/alg.png"
    malicious_sample = "./output_dataset/malicious/ab22ea22e3ae447efe01d7de5f83787c78778e31b9cadccb17b8fd784f8a9fc8.png"



if __name__ == '__main__':
    main()
