import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, model_from_json



with tf.device('/device:GPU:1'):
	


	dir_image = '/home/sharmasp/repos/ms-project/created_img/128'
	dir_csv = '/home/sharmasp/repos/ms-project/csvfiles/train'


	dict_label = {'T1':[1,1],
	                    'T2':[0,0],
	                    'T3':[1,0],
	                    'T4':[0,0],
	                    'T5':[1,0], 
	                    'T6':[1,0], 
	                    'T7':[1,0], 
	                    'T8':[1,0]}


	list_csv_files = os.listdir(dir_csv)

	list_au_label = []
	list_task_label = []
	#list_image_data = []
	#list_image_path = []

	for i  in list_csv_files:
	    if i.endswith('.csv'):

	        dataframe_au_occ = pd.read_csv(dir_csv+'/'+i)
	        frames = list(dataframe_au_occ.iloc[:,0])
	        label_au = dataframe_au_occ.iloc[:,1:].to_numpy()
	        subject, task = i.split('_')
	        task = str(os.path.splitext(task)[0])
	        for j in range(len(frames)):
	            if(os.path.isfile(dir_image+'/'+subject+'/' +task+'/'+str(frames[j]).zfill(4)+'.png')):
	                # image = cv2.imread(dir_image+'/'+subject+'/' +task+'/'+str(frames[j]).zfill(4)+'.png',0)/255
	                label = task
	                au = list(label_au[j])
	                # list_image_data.append(image)
	                list_au_label.append(au)
	                list_task_label.append(dict_label[task])

	            else:
	                pass
	    else:
	        print(i,'not csv')


	# noise = tf.random.uniform((len(list_au_label),13), minval = 0.5, maxval = 1, seed =1)
	AU_input = np.array(list_au_label).reshape(-1,1,13).astype('float32')
	noise = tf.random.uniform((len(list_au_label),13), minval = 0.5, maxval = 1, seed =1)
	# AU_and_noise = np.array(list_au_label*noise)
	AU_and_noise = np.array(list_au_label*noise).reshape(-1,1,13)
	list_task_label = np.array(list_task_label).astype('float32').reshape(-1,1,2)


	input_au_noise = keras.Input(shape =(1,13), name = 'AU_occ_input_with_noise')
	func_input = keras.Input(shape = (1,13), name = 'AU_input')
	# input_au_noise = keras.Input(shape =(13), name = 'AU_occ_input_with_noise')
	x = layers.Dense(13)(input_au_noise)
	x = layers.LeakyReLU()(x)
	x = layers.Dense(13)(x)
	x = layers.LeakyReLU()(x)
	x = layers.Dense(13)(x)
	x = layers.LeakyReLU()(x)
	x = layers.Dense(13, activation='sigmoid')(x)
	corrected_au_values = layers.Multiply()([x, func_input])
	classification_layer = layers.Dense(2, activation='sigmoid')(corrected_au_values)
	vals = Model(inputs = (input_au_noise,func_input), outputs = corrected_au_values)
	classification_model = keras.Model(inputs = (input_au_noise,func_input), outputs =classification_layer)


	# loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels=list_task_label, logits=classification_layer.numpy())
	classification_model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')  
	# classification_model.compile(optimizer = 'adadelta', loss = 'categoricalCE')
	classification_model.fit((AU_and_noise, AU_input) , list_task_label, epochs=30, batch_size=2)

	classification_model.save_weights('myweightsforclassification', overwrite=True)