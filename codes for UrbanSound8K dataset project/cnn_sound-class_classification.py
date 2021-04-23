# with the features for cnn, now its time to make the model and to the cool stuffs!!
# at the end of the code at line 119-120 a CNN model will be created and we will use that model for
# predicting and evaluating etc. 


import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

#convert the data and labels for understandable numerical data
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from datetime import datetime

df = pd.read_hdf('../features_from_UrbanSound_for_cnn.h5', 'df')


def prepare_dataset(test_size, validation_size): 	
	#load data
	X = np.array(df.feature.tolist())
	y = np.array(df.class_label.tolist())
	
	le = LabelEncoder()
	y = to_categorical(le.fit_transform(y))
	
	#create train/test split
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)
	
	#create train/validation split
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_size)

	#for CNN tensor flow expects a 3d array -->(130,13,1)
	X_train = X_train[...,np.newaxis] #4d array --> (num_samples, 130, 13,1)
	X_validation = X_validation[...,np.newaxis]
	X_test = X_test[...,np.newaxis]
	
	return X_train, X_validation, X_test, y_train, y_validation, y_test
	
def build_model(input_shape):
	
	#create model 
	model = keras.Sequential()
	#1st conv layer
	model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape= input_shape))
	model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
	model.add(keras.layers.BatchNormalization()) # speed up korbe>> normalise korbe...details bole nai...beginning level e na janleo ok!
	
	
	#param>>filters, kernel_size, 
	#2nd conv layer
	model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape= input_shape))
	#param>> kernel_size, padding same means zero padding
	model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
	model.add(keras.layers.BatchNormalization())
	
	
	#3rd conv layer
	model.add(keras.layers.Conv2D(32, (2, 2), activation = 'relu', input_shape= input_shape))
	model.add(keras.layers.MaxPool2D((2,2), strides = (2,2), padding = 'same'))
	model.add(keras.layers.BatchNormalization())
	
	
	#flatten the output and feed it into dense layer
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation = 'relu'))
	model.add(keras.layers.Dropout(0.3))
	
	#output layer
	model.add(keras.layers.Dense(10, activation = 'softmax'))

	return model


def predict(model, X, y):
	
	X = X[np.newaxis, ...]
	
	#prediction is a 2d array prediction = [[0.1, 0.2, ...]]	
	prediction = model.predict(X) #X - > (1, 130, 13, 1)
	
	# extract index with max value
	predicted_index = np.argmax(prediction, axis=1) # 1d array >> index >> [0-9]
	
	print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


def predict2(model, X, y):
	
	X = X[np.newaxis, ...]
	
	

if __name__ == '__main__':
	#creat train, validation and test sets
	X_train, X_validation, X_test, y_train, y_validation, y_test= prepare_dataset(0.25, 0.2)
	
	print(y_test[100])
	print(y_test.shape)
	#print(X_train.shape)
	#print(X_test.shape)
	
	
	#build the CNN net
	input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3] ) 
	model = build_model(input_shape)
	
	#Compile the CNN
	optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
	model.compile(optimizer=optimizer, loss= "categorical_crossentropy",
	metrics=['accuracy'])
	
	model.summary()

	#train the CNN
	start_time = datetime.now()
	model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size = 32, epochs = 100 )
	duration = datetime.now() - start_time
	print("Duration for the train {}".format(duration))
	model.save('cnn_model_after_training.h5')
	
	#evaluate the CNN on the test set
	test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 1)
	print("Accuracy on test set is : {}". format(test_accuracy))
	
	'''
	#make prediction on a sample
	X = X_test[100]
	y = y_test[100]
	predict(model, X, y)	
	'''
