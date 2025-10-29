# this is the code of the sound of AI - episode 12 >> preparing the dataset  that he will use for the next vdo to train 
# I am trying to copy the code that he used>> so that I have a practice by myself>>i m not sure this is a good way for learning or not



import json 
import os
import math
import librosa

DATASET_PATH = '/home/safat/python_code/audio/sample_of_urbansound/'
JSON_PATH = 'sample_data_10.json'
SAMPLE_RATE = 22050
#TRACK_DURATION = >>> if all the audio has same duration this will be uncommented
#SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION >> this will be also uncommented then


def save_mfcc(dataset_path, json_path, num_mfcc = 13 , n_fft = 2048, hop_length=512 , num_segments=1):
	"""Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
		:param dataset_path (str): Path to dataset
		:param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
    	:return:
    """
    #dictionary to store mapping, labels, and MFCCs
	data ={
	"mapping" : [], #fold1/fold2
	"labels": [], # label are the target  0>fold1, 1>fold2 
	"mfcc" : [] #mfcc for each segment [[],[],[]] >>> mfcc are the training input
	}
		
	#samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
	#num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
	
	
	
	
		    
	#loop through all the sub folder
	for i , (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
		# -> glob module diye erokom kisu ekta dekhsilam>> dekha lagbe abar
		#firstly we need to ensure we are not in the data set path because >>
		if dirpath is not dataset_path:
			#save genre label ( folder name ) in the mapping 
			semantic_label = dirpath.split("/")[-1]
			data["mapping"].append(semantic_label)
			print("\nProcessing: {}".format(semantic_label))
			
			
			#process all audio files in genre sub-dir
			for f in filenames:
				
				try:
					#load audio file 
					file_path = os.path.join(dirpath, f)
					signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)
				
					#line 14,15 issue for my problem
					TRACK_DURATION = math.ceil(sample_rate / 22050)
					SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
					
					#line 35,36 issue >> same case
					samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
					num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)			
					
					#process all segments of audio file -> to make the audio file segmented for making huge dataset
					#i will keep this bt make this portion commented because the dataset that I am working is enough to compute DL because I have total around 8K dataset>>\
					for d in range(num_segments):
						#calculate start and finish sample for current segment
						start = samples_per_segment * d  # d=0 ->0 d>> current sample we are in
						finish = start + samples_per_segment #d=0 -> num samples per segment
						
						
						#extract mfcc
						mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft = n_fft , hop_length = hop_length)
						mfcc = mfcc.T #transpose matrix
						#store only mfcc feature with expected number of vectors
						#ekhane onek aaje baaje value ashbe >>> taai exact gula niye kaaj kora lagbe
						
						print("{} length >> num_mfcc_per_segment {} ".format(len(mfcc), num_mfcc_vectors_per_segment))
						if len(mfcc) == num_mfcc_vectors_per_segment: # -> expected num of mfcc per segment
							data["mfcc"].append(mfcc.tolist())
							data['labels'].append(i-1)
							print("{}, segment:{}".format(file_path, d+1))
							
						
				except Exception:
					pass		
    #save MFCCs to Json file
	with open(json_path, 'w') as fp:
		json.dump(data, fp, indent=4)
        	
if __name__ == '__main__':
	save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1)
