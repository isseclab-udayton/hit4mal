from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
import os
from shutil import copyfile
import shutil
import sys
from keras.models import model_from_json
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

img_width, img_height = 150, 150
train_data_dir = 'train/'
validation_data_dir = 'validation/'
nb_train_samples = 4000
nb_validation_samples = 2000
epochs = 100 
batch_size = 16

def create_dataset(main_path, ratio):
    #main_path = 'malicious'
    train_path = os.getcwd() + '/train/' + main_path
    val_path = os.getcwd() + '/validation/' + main_path

    # check path exists
    if (not os.path.exists(main_path)):
        print("dataset not found")
        sys.exit(0)
    else:
        # delete existing files
  
        if(os.path.exists(train_path)):
            shutil.rmtree(train_path)
        os.makedirs(train_path)


        if(os.path.exists(val_path)):
            shutil.rmtree(val_path)
        os.makedirs(val_path)


        main_files = [os.path.join(main_path, f) for f in os.listdir(main_path)]
        # copy train files
        for t in main_files[0:ratio]:
            t_base = os.path.basename(t)
            copyfile(t, train_path + t_base)
        # copy test files
        for v in main_files[ratio:]:
            v_base = os.path.basename(v)
            copyfile(v, val_path +  v_base)
        print("OK")

# SVM 
def svc(traindata, trainlabel, testdata, testlabel):
  print("Start training SVM...")
  svcClf = SVC(C=1.0, kernel="rbf", cache_size = 3000)
  svcClf.fit(traindata, trainlabel)

  pred_testlabel = svcClf.predict(testdata)
  num = len(pred_testlabel)
  accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
  print("cnn-svm Accuracy:", accuracy)


def create_model(input_shape):
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
    #model.add(Dense(64))
    #model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def VGG_16(input_shape):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(input_shape)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def save_model(model):
    # save model layout
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
      json_file.write(model_json)

    # save model weight
    model.save_weights("model/model.h5")
    print("Saved model to disk")

def load_model():
    # read model layout
    json_file = open("model/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weghts into new model
    loaded_model.load_weights("model/model.h5")
    return loaded_model


def main():
    malicious_path = 'malicious/'
    benign_path = 'benign/'

    print("Loading malcious samples")
    create_dataset(malicious_path, 2000)
    print("Loading benign samples")
    create_dataset(benign_path, 2000)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # create model
    model = create_model(input_shape)
    #model = VGG_16(input_shape)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

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

    tb = TensorBoard(log_dir='tensorboard/')
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size, callbacks = [tb])

    # model.save_weights('first_try.h5')
    # save model
    save_model(model)

    # load model
    loaded_model = load_model()

    # summary of model
    loaded_model.summary()

    # get intermediate output layer
    # get_3rd_layer_output = K.function([loaded_model.layers[0].input, K.learning_phase()],
    #                                   [loaded_model.layers[2].output])
    # layer_output = get_3rd_layer_output([train_generator, 0])[0]
    # print(layer_output.shape)

if __name__ == "__main__":
    main()



