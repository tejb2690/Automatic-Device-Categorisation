

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import ImageFile
import random
from tqdm import tqdm
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Cropping2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
#from keras.layers.convolutional import Cropping2D
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers import LSTM
from keras.layers import GaussianNoise
from keras import optimizers



#from pydatset.cifar10 import get_CIFAR10_data

random.seed(340958234985)
np.random.seed(2093846)
image_names = [item.replace('C:/Users/tejb/Desktop/newDataOrig10CatTest/', '') 
for item in sorted(glob("C:/Users/tejb/Desktop/newDataOrig10CatTest/*/"))]
#image_names = [item.replace('C:/Users/tejb/Desktop/GrayDataText/', '') for item in sorted(glob("C:/Users/tejb/Desktop/GrayDataText/*/"))]

number_of_image_categories = len(image_names)
print('%d image categories.' % number_of_image_categories)
print('ALL categories:')
print(image_names[:7])

def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), number_of_image_categories)
    return image_files, image_targets


image_files, image_targets = load_dataset('C:/Users/tejb/Desktop/newDataOrig10CatTest')
#image_files, image_targets = load_dataset('C:/Users/tejb/Desktop/GrayDataText')


trains_validate_files, test_files, trains_validate_targets, test_targets = \
    train_test_split(image_files, image_targets, test_size=0.2, random_state=42)

train_files, valid_files, train_targets, valid_targets = \
    train_test_split(trains_validate_files, trains_validate_targets, test_size=0.25, random_state=42)
    





image_names = [item[20:-1] for item in sorted(glob("C:/Users/tejb/Desktop/newDataOrig10CatTest/*/"))]
#image_names = [item[20:-1] for item in sorted(glob("C:/Users/tejb/Desktop/GrayDataText/*/"))]


print('%s images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('%d training images.' % len(train_files))
print('%d validation images.' % len(valid_files))
print('%d test images.'% len(test_files))

####code to add augmentation like shrink image and mask with background of original shape then shift left right bottom and top




def path_to_tensor(img_path):

    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)

    return np.expand_dims(img_array, axis=0)

def paths_to_tensor(img_paths):
    
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
#######resize image to 28x28 for ZCA_whitening purpose

def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop 48x48px
    desired_width, desired_height = 64, 64

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((64, 64))

    img = image.img_to_array(img)
    return img / 255.

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    #rescale=1./255,
    preprocessing_function=preprocess,
    #samplewise_center=True,
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zca_whitening=True,
    #rescale=1./255,
    fill_mode='nearest',
    zoom_range=[0.9, 1.25],
    #zoom_range=0.2,
    brightness_range=[1.0, 1.9]
    #featurewise_std_normalization=True,
)
'''
valid_datagen = ImageDataGenerator(rescale = 1./255,n
    width_shift_range=0.4,
    height_shift_range=0.4,
)
'''
#test_datagen = ImageDataGenerator(rescale = 1./255)
#print("train_tensors shape before Datagen")
#print(train_tensors.shape)
#train_datagen.fit(image_files)
datagen.fit(train_tensors)
#valid_datagen.fit(valid_tensors)

print("train_tensors shape after Datagen")
print(train_tensors.shape)
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''
for X_, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break
'''



model = Sequential()
### adding the cropping layer
#model.add(Cropping2D(cropping=((2, 2), (4, 4)),
#                         input_shape=(128, 128, 3)))
#model.add(Cropping2D(cropping=((40, 40), (40, 40)),
#                         input_shape=(128, 128, 3),data_format='channels_last'))
#model.add(Conv2D(filters=4, kernel_size=2, padding='same',
#                 activation='relu', input_shape=(64, 64, 3),data_format='channels_last'))
model.add(Conv2D(filters=8, kernel_size=3, padding='same',
                 activation='relu', input_shape=(64, 64, 3),data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#model.add(GaussianNoise(0.01))
#model.add(LSTM(128))
#model.add(Dropout(0.1))
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
###Add Gaussian noise layer
#model.add(GaussianNoise(0.01))
#model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Conv2D(filters=24, kernel_size=3, padding='same',activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=2))
###Add Gaussian noise layer
#model.add(GaussianNoise(0.01))
#model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=3, padding='same',activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
###Add Gaussian noise layer
#model.add(GaussianNoise(0.01))
#model.add(LSTM(128))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu', input_shape=(64, 64, 3)))
#model.add(Dense(256, activation='relu',activity_regularizer=l1(0.0001)))

###Add Gaussian noise layer
#model.add(GaussianNoise(0.01))
model.add(Dropout(0.4))

#model.add(Dense(7, activation='softmax',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model.add(Dense(7, activation='softmax', input_shape=(64, 64, 3)))
#model.add(Dense(10, activation='softmax',kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))


model.summary()
#sgd=optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
#model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#opt optimizer9
epochs = 50
batch_size=10
model_json = model.to_json()
#with open("C:/Users/tejb/Desktop/GrayDataText/weights.best.grayimage_classifier.json", "w") as json_file:
with open("C:/Users/tejb/Desktop/newDataOrig10CatTest/weights.best.image_classifier_newDataOrig10CatTest4_final64x6410cat.json", "w") as json_file:
    json_file.write(model_json)


#checkpointer = ModelCheckpoint(filepath='C:/Users/tejb/Desktop/GrayDataText/weights.best.grayimage_classifier.hdf5',
#                               verbose=1, save_best_only=True)


checkpointer = ModelCheckpoint(filepath='C:/Users/tejb/Desktop/newDataOrig10CatTest/weights.best.image_classifier_newDataOrig10CatTest4_final64x6410cat.hdf5',
                               verbose=1, save_best_only=True)

model.fit_generator(datagen.flow(train_tensors, train_targets,batch_size=batch_size), validation_data=(valid_tensors, valid_targets),
           steps_per_epoch=train_tensors.shape[0] // batch_size,epochs=epochs, callbacks=[checkpointer], verbose=1)
print("calculate train_tensors size")
print(train_tensors.shape)
#model.load_weights('C:/Users/tejb/Desktop/GrayDataText/weights.best.grayimage_classifier.hdf5')
#model.load_weights('C:/Users/tejb/Desktop/newDataOrig10CatTest/weights.best.image_classifier.hdf5')

predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in train_tensors]
train_accuracy = 100*np.sum(np.array(predictions)==np.argmax(train_targets, axis=1))/len(predictions)
print('Train accuracy: %.4f%%' % train_accuracy)
predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
###### evaluating confusion matrix
Y_test = np.argmax(test_targets, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(test_tensors)
print(classification_report(Y_test, y_pred))

