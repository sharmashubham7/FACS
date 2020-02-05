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
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')


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
list_image_path = []

for i  in list_csv_files:

    if i.endswith('.csv'):


        dataframe_au_occ = pd.read_csv(dir_csv+'/'+i)
        frames = list(dataframe_au_occ.iloc[:,0])
        label_au = dataframe_au_occ.iloc[:,1:].to_numpy()
        subject, task = i.split('_')
        task = str(os.path.splitext(task)[0])
        for j in range(len(frames)):
            if(os.path.isfile(dir_image+'/'+subject+'/' +task+'/'+str(frames[j]).zfill(4)+'.png')):
                # image = cv2.imread(dir_image+'/'+subject+'/' +task+'/'+str(frames[j]).zfill(4)+'.png',0)/255.0
                # image = cv2.resize(image, (128,128))
                label = task
                au = list(label_au[j])
                pathx = dir_image+'/'+subject+'/' +task+'/'+str(frames[j]).zfill(4)+'.png'

                list_image_path.append(pathx)
                list_au_label.append(au)
                list_task_label.append(task)
            else:
                pass

            
    #else:
        #print(i,'not csv')








AU_input = np.array(list_au_label).reshape(-1,1,13).astype('float32')
noise = tf.random.uniform((len(list_au_label),13), minval = 0.5, maxval = 1, seed =1)
AU_and_noise = np.array(list_au_label*noise).reshape(-1,1,13)


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


classification_model.load_weights('myweightsforclassification')


AU_inputx = vals.predict((AU_and_noise,AU_input))


latent_dim = 50
BATCH_SIZE = 12
epochs = 30

AU_inputxx = tf.data.Dataset.from_tensor_slices(AU_inputx.reshape(-1,1,13).astype('float32'))

image_path_data = tf.data.Dataset.from_tensor_slices(np.array(list_image_path).reshape(-1,1))

total_input = tf.data.Dataset.zip((image_path_data, AU_inputxx)).batch(BATCH_SIZE).shuffle(1000)



image_input = keras.Input(shape = (128,128,1), name = 'image_input')
func_input = keras.Input(shape = (1,13), name = 'AU_input')


x = layers.Conv2D(filters=32, kernel_size=5, strides=(2,2), padding='same')(image_input)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=64, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=256, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

img_encoded = layers.Dense(2*latent_dim, name = 'encoded_image')(x)

model_encoder = keras.Model(inputs = image_input, outputs = img_encoded)

discriminator_input = keras.Input(shape = (2*latent_dim), name = 'disc_input')

x = layers.LeakyReLU()(discriminator_input)

x = layers.Dense(latent_dim)(x)
x = layers.LeakyReLU()(x)

x = layers.Dense(latent_dim)(x)
x = layers.LeakyReLU()(x)

x = layers.Dense(latent_dim/2)(x)
x = layers.LeakyReLU()(x)
disc_output = layers.Dense(1,  name= 'disc_output')(x)

discriminator_for_normalization = keras.Model(inputs = discriminator_input, outputs = disc_output)


flattened_au = layers.Flatten()(func_input)
combined_input = layers.concatenate([img_encoded, flattened_au])


x = layers.Dense(1024)(combined_input)

x = layers.Dense(1024)(x)

x = layers.Dense(16384)(x)
x = layers.Reshape((8,8,256))(x)
x = layers.Conv2DTranspose(filters=256, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(2,2), padding='same')(x)
x = layers.LeakyReLU()(x)

output_img= layers.Conv2DTranspose(filters = 1, kernel_size =5, strides= (1,1), padding = 'same', name = 'output_image')(x)

#model_decoder = keras.Model(inputs = combined_input, outputs = output_img )

total_discriminator_input = keras.Input(shape = (128,128,1), name= 'image_input_for_disc')
x = layers.LeakyReLU()(total_discriminator_input)

x = layers.Flatten()(x)
x = layers.LeakyReLU()(x)
x = layers.Dense(latent_dim)(x)
x = layers.LeakyReLU()(x)

x = layers.Dense(latent_dim)(x)
x = layers.LeakyReLU()(x)

x = layers.Dense(latent_dim/2)(x)
x = layers.LeakyReLU()(x)
disc_output_real_or_fake = layers.Dense(1,  name= 'disc_output_real_fake')(x)

total_discriminator = keras.Model(inputs = total_discriminator_input, outputs = disc_output_real_or_fake)



model_AE = keras.Model(inputs = [image_input, func_input], outputs = output_img)







model_AE.load_weights('AE_weights')
total_discriminator.load_weights('totat_disc_weights')
discriminator_for_normalization.load_weights('disc_norm')




cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_func(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
AE_optimizer = tf.keras.optimizers.Adam(1e-4)
encoder_optimizer = tf.keras.optimizers.Adam(1e-6)
total_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
AE_total_optimizer = tf.keras.optimizers.Adam(1e-6)

def encoder_loss_func(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def AE_loss_func(images, output_AE):
    return tf.keras.losses.MSE(images, output_AE)


def total_disc_loss_func(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def AE_total_disc_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    



def train_step(images, AU_input):
    with tf.GradientTape() as AE_tape, tf.GradientTape() as encoder_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as total_disc_tape, tf.GradientTape() as AE_total_tape :
        
        
        output_image = model_AE([images, AU_input])
        loss = tf.keras.losses.MSE(images, output_image)
        gradientsofAE = AE_tape.gradient(loss, model_AE.trainable_variables)
        AE_optimizer.apply_gradients(zip(gradientsofAE,model_AE.trainable_variables ))


        images_from_encoder = model_encoder(images)
        fake_output = discriminator_for_normalization(images_from_encoder, training = True)
        loss_encoder = encoder_loss_func(fake_output)


        noise = tf.random.normal([BATCH_SIZE,2*latent_dim])##### actually noise
        real_output = discriminator_for_normalization(noise, training = True)
        fake_output = discriminator_for_normalization(images_from_encoder, training = True)
        loss_encoder = encoder_loss_func(fake_output)
        gradients_of_encoder = encoder_tape.gradient(loss_encoder, model_encoder.trainable_variables)

        loss_discriminator = discriminator_loss_func(real_output, fake_output)
        #print(loss_discriminator)
        gradients_of_discriminator = disc_tape.gradient(loss_discriminator, discriminator_for_normalization.trainable_variables)

        encoder_optimizer.apply_gradients(zip(gradients_of_encoder, model_encoder.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_for_normalization.trainable_variables))

        
        
        total_real_output = total_discriminator(images,  training = True)
        total_fake_output = total_discriminator(output_image,  training = True)
        
        #print('##########')
        #print(total_real_output,"###")
        #print(total_fake_output,'***')
            
        loss_total_discriminator = total_disc_loss_func(total_real_output, total_fake_output)
        gradients_of_total_discriminator = total_disc_tape.gradient(loss_total_discriminator, total_discriminator.trainable_variables)
        total_discriminator_optimizer.apply_gradients(zip(gradients_of_total_discriminator, total_discriminator.trainable_variables))
        
        
        loss_AE_total_disc = AE_total_disc_loss(total_fake_output)
        gradients_of_AE_total_disc = AE_total_tape.gradient(loss_AE_total_disc, model_AE.trainable_variables)
        AE_total_optimizer.apply_gradients(zip(gradients_of_AE_total_disc, model_AE.trainable_variables))
        
   
        
def image_loader(paths):
    loaded_image = []
    for path in paths:
        #print(np.array(z).astype(str))
        path = np.array(path).astype(str)
        #print(str(k))
        image = cv2.imread(str(path),0)/255
        #image = cv2.resize(image,(128,128))
        loaded_image.append(image)
        # print(image)
    return np.array(loaded_image)


def train(total_inputx):
    for epoch in range(epochs):
    	if((epoch+1)%5==0):
    		model_AE.save_weights('AE_weights',overwrite=True)
    		total_discriminator.save_weights('totat_disc_weights', overwrite=True)
    		discriminator_for_normalization.save_weights('disc_norm', overwrite=True)
    	start = time.time()
    	print("starting epoch:" ,epoch)
    	for batch_input in total_inputx:
    		image_data, AU_inputs = batch_input
    		image_datax = tf.map_fn(image_loader, image_data, dtype = tf.float64)
    		image_datax = np.array(image_datax).reshape(-1,128,128,1)
    		AU_inputx = np.array(AU_inputs).reshape(-1,1,13).astype('float32')
    		train_step(image_datax, AU_inputx)








    	print('time for epoch {} is {} sec'.format(epoch, time.time()-start))
		
		# if (((epoch+1)%5) == 0):

		# 	model_AE.save_weights('AE_weights', overwrite = True)
		# 	total_discriminator.save_weights('totat_disc_weights', overwrite =True)
		# 	discriminator_for_normalization.save_weights('disc_norm', overwrite = True)

    	
    print('finished')



train(total_input)



model_AE.save_weights('AE_weights', overwrite = True)
total_discriminator.save_weights('totat_disc_weights', overwrite =True)
discriminator_for_normalization.save_weights('disc_norm', overwrite = True)






