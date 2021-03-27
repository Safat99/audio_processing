import numpy as np
import pandas as pd
import librosa
import os
from datetime import datetime

max_pad_len = 174
full_dataset_path = os.path.abspath('../UrbanSound8K/audio/')
metadata = pd.read_csv('../UrbanSound8K/metadata/UrbanSound8K.csv')

features = []

def extract_feature(file):

	try:	
		audio, sr = librosa.load(file, res_type = 'kaiser_fast')
		mfccs = librosa.feature.mfcc(audio, sr = sr, n_mfcc =40)
		pad_width = max_pad_len - mfccs.shape[1]
		mfccs = np.pad(mfccs, pad_width =((0,0), (0,pad_width)), mode = 'constant')
	
	except Exception as e:
		print("Error happened while parsing the file", file)
		return None
		
	return mfccs
	
def extract_feature_all():
	
	start_time = datetime.now()
	for index, row in metadata.iterrows():
		audio_file_name = os.path.join(os.path.abspath(full_dataset_path) , 'fold' + str(row['fold']) + '/' + str(row['slice_file_name']))
		
		class_label = row['class_name']
		data = extract_feature(audio_file_name)
		
		features.append([data, class_label])
		print("loaded {} file ".format(index))
	
	featuresdf = pd.DataFrame(features, columns = ['feature', 'class_label'])
	loaded_time = datetime.now() - start_time
	print("total {} files loaded by taking {} time".format(len(featuresdf), loaded_time))
	
	return featuresdf

if __name__ == '__main__':

	df = extract_feature_all()
	df.to_hdf('features_from_UrbanSound_for_cnn.h5', key='df',mode='w')
