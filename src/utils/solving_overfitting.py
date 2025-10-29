import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATASET_PATH = '/home/safat/python_code/audio/sample_data_10.json'

#load sata
def load_data(dataset_path):
	with open(dataset_path, "r") as fp:
		data = json.load(fp)
	
	#convert lists into numpy arrays >> labels and mfccs are lists at the json
	inputs = np.array(data["mfcc"])
	targets = np.array(data["labels"])
	
	print("Data successfully loaded!")
	
	return inputs,targets


def plot_history(history):
	
	fig, axs = plt.subplots(2)
	
	#create accuracy subplots
	axs[0].plot(history.history["accuracy"], label="train accuracy")
	axs[0].plot(history.history["val_accuracy"], label="test accuracy")
	axs[0].set_ylabel("Accuracy")
	axs[0].legend(loc="lower right")
	axs[0].set_title("Accuracy eval")
	
	#create error subplots
	axs[1].plot(history.history["loss"], label="train error")
	axs[1].plot(history.history["val_loss"], label="test error")
	axs[1].set_ylabel("Error")
	axs[1].set_xlabel("Epoch")
	axs[1].legend(loc="upper right")
	axs[1].set_title("Error eval")

	plt.show()
	

	
if __name__ == '__main__':
	inputs, targets = load_data(DATASET_PATH)
	
	#split the data into train and test
	inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,targets, test_size = 0.3)
	 
	
	#build the network architecture
	#we need tf	
	model = keras.Sequential([
		#for mlp fully connected dense layer
		
		#input layer
		keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),#this 2 dimensional array will be flattened >>
		#each track have mfccs...each mfcc vectors are taken at a certain interval >> hop length >>
		# frst dimension given by the interval, secnd dmnsions are the values for that interval
		
		#1st hidden layer 
		keras.layers.Dense(512, activation = "relu"),
		keras.layers.Dropout(0.3),
		#advantage of relu instead of sigmoid >> 1) better convergence 2) reduced likelihood of vanishing gradient
		#sigmoid ke differntiate korle gradient kome >> kintu 0 hoy na >> gradient vanishing eitakei bole
		
		#2nd hidden layer 
		keras.layers.Dense(256, activation = "relu"),
		keras.layers.Dropout(0.3),
		#3rd hidden layer 
		keras.layers.Dense(64, activation = "relu"),
		keras.layers.Dropout(0.3),
		#layer gula arbitary but output e 10 ta karon 10 ta genre classify korsilo oy
		
		# output layer 
		keras.layers.Dense(10, activation = "softmax")
		#softmax function basically normalise the output for us.. the sum of all this 10 predictions = 1>>> 
		# we pick the neuron that has the highest value 
		
	])
	
	#compile network
	optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
	#adam is an extension of stchochastic gradient descent  
	model.compile(optimizer = optimizer,
					loss="sparse_categorical_crossentropy",
					metrics = ["accuracy"])
					#since this MLP we have used sparsed....fucntion
					
					
					
	model.summary()#which gives a report 
	
	#train network
	history = model.fit(inputs_train, targets_train,
				validation_data=(inputs_test, targets_test),
				epochs=100,
				batch_size = 32)
	
	#plot accuracy amd error over the epochs
	plot_history(history)
