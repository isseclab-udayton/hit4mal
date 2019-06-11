from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
from shutil import copyfile
import shutil

benign_list = []
malicious_list = []
# def get_path_list(root_dir):
#   path_list = []
#   for subdir, dirs, files in os.walk(root_dir):
#       for file in files:
#           path_list.append(os.path.join(subdir, file))
#     #return path_list

# dim\ensisons of our images
img_width, img_height = 256, 1024
dataset = '/home/worker1/Ly/PackDeep/dataset/images/'
train_val = '/home/worker1/Ly/PackDeep/'

# # benign_samples = get_path_list(dataset + 'benign')
# for subdir, dirs, files in os.walk(dataset + 'benign'):
#     for file in files:
#         benign_list.append(os.path.join(subdir, file))
# print(len(benign_list))


# benign_train = benign_list[:len(benign_list)/2]

# for bt in benign_train:
#   copyfile(bt, train_val + 'train/benign/'+ os.path.basename(bt))

# benign_val = benign_list[len(benign_list)/2:]


# for subdir, dirs, files in os.walk(dataset + 'malicious'):
#     for file in files:
#         malicious_list.append(os.path.join(subdir, file))
# print(len(malicious_list))

# malicious_train = malicious_list[:len(malicious_list)/2]
# malicious_val = malicious_list[len(malicious_list)/2:]


N = 1000


def move_malicious(abs_dirname, ratio):
    """Move files into subdirectories."""
    train_malicious = 'train/malicious/'
    shutil.rmtree(train_malicious)
    os.makedirs(train_malicious)
    val_malicious = 'validation/malicious/'
    shutil.rmtree(val_malicious)
    os.makedirs(val_malicious)
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    for f in files[0:ratio]:
        # create new subdir if necessary
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
        #os.mkdir('train/' + file_type)
            #curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)

        copyfile(f, train_malicious + f_base)

    for f in files[ratio:]:
        # create new subdir if necessary
        #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
        #os.mkdir('validation/' + file_type )
        f_base = os.path.basename(f)

        copyfile(f, val_malicious + f_base)
        #copyfile(f, 'validation/malicious/' + f_base)


def move_benign(abs_dirname, ratio):
    """Move files into subdirectories."""

    train_benign = 'train/benign/'
    shutil.rmtree(train_benign)
    os.makedirs(train_benign)
    val_benign = 'validation/benign/'
    shutil.rmtree(val_benign)
    os.makedirs(val_benign)
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    for f in files[0:ratio]:
        # create new subdir if necessary
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
        #os.mkdir('train/' + file_type)
            #curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)

        copyfile(f, train_benign + f_base)
        #copyfile(f, 'train/benign/' + f_base)

    for f in files[ratio:]:
        # create new subdir if necessary
        #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
        #os.mkdir('validation/' + file_type )
        f_base = os.path.basename(f)

        copyfile(f, val_benign + f_base)
        #copyfile(f, 'validation/benign/' + f_base)

move_malicious('malicious', 2000)
move_benign('benign', 2000)
# Divide dataset

train_data_dir = '/home/worker1/Ly/PackDeep/train'
validation_data_dir = '/home/worker1/Ly/PackDeep/validation'
# train_data_dir = '/home/lyvd/GitHub/PackDeep/images/train'
# validation_data_dir = 'images/validation'
nb_train_samples = 4000
nb_validation_samples = 2000
epochs = 100
batch_size = 16

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
              optimizer='rmsprop', metrics=['accuracy'])

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# # This is the augmentation configuration we will use for testing
# # only rescaling
# test_datagen = ImageDataGenerator(rescale = 1. /255)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'validation',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# model.save_weights('first_try.h5')
# This is a PIL image
# img = load_img('/home/lyvd/GitHub/PackDeep/DC/train_set/cats/cat.0.jpg')
# x = img_to_array(img) # This is a Numpy array with share (3, 1510. 150)
# print(x.shape)
# x= x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3,
# 150, 150)

# # The .flow() command below generates batches of randomly transfomed images
# # and saves the results to `previwew` directory
# i = 0
# for batch in datagen.flow(x, batch_size = 1, save_to_dir='preview', save_prefix='arp', save_format = 'jpeg'):
#   i += 1
#   if i > 20:
#       break
