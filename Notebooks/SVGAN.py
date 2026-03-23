from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
import h5py	
import tensorflow as tf 
print(tf.config.list_physical_devices('GPU'))

latent_dim = 200
n_samples =600 #multiple of 6
lr=0.0002
n_epochs=1
n_batch=200
location="runs\\genfile\\"

# define the standalone generator model
def define_generator(latent_dim):

	in_lat = Input(shape=(latent_dim,))
	#Start with enough dense nodes to be reshaped and ConvTransposed to 28x28x1
	n_nodes = 256 * 16 * 16
	X = Dense(n_nodes)(in_lat)
	X = LeakyReLU(alpha=0.2)(X)
	X = Reshape((16, 16, 256))(X)

	X = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(X) #32x32x128
	X = LeakyReLU(alpha=0.2)(X)

	X = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(X) #32x32x64
	X = LeakyReLU(alpha=0.2)(X)
	# output
	out_layer = Conv2DTranspose(1, (3,3), strides=(2,2), activation='tanh',
                             padding='same')(X) #64x64x1
	# define model
	model = Model(in_lat, out_layer)
	return model

# gen_model=define_generator(100)
# print(gen_model.summary())

# define the base discriminator model for sup and unsup discriminators
#This is the base discriminator that supervised and unsupervised are going to share weights from.
#(I know that the above ssentence is bad english!!!)
def define_discriminator(n_classes,in_shape=(64,64,1)):
    in_image = Input(shape=in_shape)
    X = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
    X = LeakyReLU(alpha=0.2)(X)

    X = Conv2D(64, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)

    X = Conv2D(128, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)

    X = Flatten()(X)
    X = Dropout(0.4)(X) #Consider adding more dropout layers to minimize overfitting - remember we work with limited labeled data.
    X = Dense((len(n_classes)+2))(X)

    model = Model(inputs=in_image, outputs=X)

    return model

#Define the supervised discriminator.
#Multiclass classification, so we will use softmax activation.
#To avoid converting our labels to categorical, we will work with sparse categorical crossentropy loss.
def define_sup_discriminator(disc,lr=0.0002):
    model=Sequential()
    model.add(disc)
    model.add(Activation('softmax'))
    #Let us use sparse categorical loss so we dont have to convert our Y to categorical
    model.compile(optimizer=Adam(learning_rate=lr, beta_1=0.5),
                  loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    return model


#Define the unsupervised discriminator
#Takes the output of the supervised, just before the softmax activation.
#Then, adds a layer with calculation of sum of exponential outputs. (defined below as custom_activation)
# Reference: https://arxiv.org/abs/1606.03498

#This custom activation layer gives a value close to 0 for smaller activations
#in the prior discriminator layer. It gives values close to 1 for large activations.
#This way it gives low activation for fake images. No need for sigmoid anymore.

# custom activation function for the unsupervised discriminator
#D(x) = Z(x) / (Z(x) + 1) where Z(x) = sum(exp(l(x))). l(x) is the output from sup discr. prior to softmax
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x /(Z_x+1)

    return D_x

#You can also try the built in tensorflow function: tf.reduce_logsumexp(class_logits, 1)

def define_unsup_discriminator(disc,lr=0.0002):
    model=Sequential()
    model.add(disc)
    model.add(Lambda(custom_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5),metrics=['accuracy'])
    return model


# disc=define_discriminator()
# disc_sup=define_sup_discriminator(disc)
# disc_unsup=define_unsup_discriminator(disc)
# print(disc_unsup.summary())


# define the combined generator and discriminator model, for updating the generator
def define_gan(gen_model, disc_unsup,lr=0.0002):

	disc_unsup.trainable = False # make unsup. discriminator not trainable
	gan_output = disc_unsup(gen_model.output) #Gen. output is the input to disc.
	model = Model(gen_model.input, gan_output)
	model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5))
	return model

# gan_model = define_gan(gen_model, disc_unsup)
# print(gan_model.summary())


#Added code for data addition =====================================================Added by Ankit
def data_extract():
	label_list={'0.0': 2, '0.1': 2,
			  '1.0': 3, '1.1': 3, '2.0': 3,'2.1': 3,
			  '3.0': 4, '3.1': 4, '3.2': 5, '3.3': 6,
			  '4.0': 4, '4.1': 4, '4.2': 5, '4.3': 6,
			  '5.0': 4, '5.1': 4, '5.2': 5, '5.3': 6}
	
	a=h5py.File("Dataset_train_28989.h5","r")
	b=h5py.File("Dataset_test_7247.h5","r")
	x1= a["image"]
	x2= b["image"]
	y1= a["label"]
	y2= b["label"]
	y1_list=list()
	y2_list=list()
	for i in range(y1.shape[0]):
		vali=(y1[i][0])+(0.1*y1[i][3])
		y1_list.append(label_list[f"{vali}"])
	for j in range(y2.shape[0]):
		valj=(y2[j][0])+(0.1*y2[j][3])
		y2_list.append(label_list[f"{valj}"])
	y_train=np.array(y1_list,dtype="float32")
	y_test=np.array(y2_list,dtype="float32")
	ysort = list(set(y1_list))
	ysort.sort()    
	data=((x1[:,18:82,48:112],y_train),(x2[:,18:82,48:112],y_test))
	return data,ysort,label_list


# load the images
def load_real_samples(data,n_classes):
    (trainX, trainy), (_, _) = data		#load_data() #uncomment this for origianl=====Added by Ankit
    X = expand_dims(trainX, axis=-1)
    #X = X.astype('float32')
    X = ((X*-1) - 127.5) / 127.5  # scale from [0,255] to [-1,1] as we will be using tanh activation.
    print(X.shape, trainy.shape)
    return [X, trainy]

#data = load_real_samples()

#select subset of the dataset for supervised training
#Let us pick only 100 samples to be used in supervised training.
#Also, we need to ensure we pick 10 samples per class to ensure a good balance
#of data between classes.
def select_supervised_samples(dataset,n_classes,n_samples=100):
	X, y = dataset
	X_list, y_list = list(), list()
	n_per_class = int(n_samples / len(n_classes)) #Number of amples per class.
	for i in n_classes:
		X_with_class = X[y == i] # get all images for this class
		ix = randint(0, len(X_with_class), n_per_class) # choose random images for each class
		[X_list.append(X_with_class[j]) for j in ix] # add to list
		[y_list.append(i) for j in ix]
	return asarray(X_list), asarray(y_list) #Returns a list of 2 numpy arrays corresponding to X and Y


# Pick real samples from the dataset.
#Return both images and corresponding labels in addition to y=1 indicating that the images are real.
#Remember that we will not use the labels for unsupervised, only used for supervised.
def generate_real_samples(dataset, n_samples):

	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix] #Select random images and corresponding labels
	y = ones((n_samples, 1)) #Label all images as 1 as these are real images. (for the discriminator training)
	return [X, labels], y

# generate latent points, to be used as inputs to the generator.
def generate_latent_points(latent_dim, n_samples):
	z_input = randn(latent_dim * n_samples)
	z_input = z_input.reshape(n_samples, latent_dim) # reshape for input to the network
	return z_input

# Generate fake images using the generator and above latent points as input to it.
#We do not care about labeles so the generator will not know anything about the labels.
def generate_fake_samples(generator, latent_dim, n_samples):

	z_input = generate_latent_points(latent_dim, n_samples)
	fake_images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1)) #Label all images as 0 as these are fake images. (for the discriminator training)
	return fake_images, y

# report accuracy and save plots & the model periodically.
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset,acc_list,n_steps, n_samples=100):
	# Generate fake images

	X, _ = generate_fake_samples(gen_model, latent_dim, n_samples)

	X = (X + 1) / 2.0 # scale to [0,1] for plotting
	# plot images
	fig = plt.figure(figsize=(100, 100))
	for i in range(25):
		plt.subplot(5, 5, 1 + i)
		plt.axis('off')
		plt.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to drive
	filename1 = f'{location}generated_plot_{step+1}.png'
	plt.savefig(filename1)
	plt.close()

	# evaluate the discriminator
	X, y = dataset
	_, acc = disc_sup.evaluate(X, y, verbose=0)
	filename2 = f'{location}gen_model_{step+1}.h5'
	filename3 = f'{location}disc_sup_{step+1}_{acc * 100 :2.2f}.h5' #filename3 = 'disc_sup_%04d.h5' % (step+1)
	print('Discriminator Accuracy: %.3f%%' % (acc * 100))
	if (acc*100)> acc_list[-1]:
		# save the generator model
		gen_model.save(filename2)
		# save the Discriminator (classifier) model
		disc_sup.save(filename3)
		print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
	elif (step+1)==n_steps:
		# save the generator model
		gen_model.save(filename2)
		# save the Discriminator (classifier) model
		disc_sup.save(filename3)
		print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
	else:
		print('>Not_Saved: %s, %s, and %s' % (filename1, filename2, filename3))
	return acc,filename3,filename2

# train the generator and discriminator
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=20, n_batch=100):

    # select supervised dataset for training.
    #Remember that we are not using all 60k images, just a subset (100 images, 10 per class. )
	X_sup, y_sup = select_supervised_samples(dataset,n_classes)
	#print(X_sup.shape, y_sup.shape)

	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# iterations
	n_steps = bat_per_epo * n_epochs

	half_batch = int(n_batch / 2)
	print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs,
                                                              n_batch, half_batch,
                                                              bat_per_epo, n_steps))

    #  enumerate epochs
	acc_list= list()
	acc_list.append(0)
	for i in range(n_steps):
		# update supervised discriminator (disc_sup) on real samples.
        #Remember that we use real labels to train as this is supervised.
        #This is the discriminator we really care about at the end.
        #Also, this is a multiclass classifier, not binary. Therefore, our y values
        #will be the real class labels for MNIST. (NOT 1 or 0 indicating real or fake.)
		[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
		sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)

		# update unsupervised discriminator (disc_unsup) - just like in our regular GAN.
        #Remember that we will not train on labels as this is unsupervised, just binary as in our regular GAN.
        #The y_real below indicates 1s telling the discriminator that these images are real.
        #do not confuse this with class labels.
        #We will discard this discriminator at the end.
		[X_real, _], y_real = generate_real_samples(dataset, half_batch) #
		d_loss_real,unsup_racc = disc_unsup.train_on_batch(X_real, y_real)
        #Now train on fake.
		X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
		d_loss_fake,unsup_facc = disc_unsup.train_on_batch(X_fake, y_fake)

		# update generator (gen) - like we do in regular GAN.
        #We can discard this model at the end as our primary goal is to train a multiclass classifier (sup. disc.)
		X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
		gan_loss = gan_model.train_on_batch(X_gan, y_gan)

		# summarize loss on this batch
		print('>%d, c-disc[%.3f,%.0f], d[real[%.3f,%.0f],fake[%.3f,%.0f]], gan[%.3f]' % (i+1, sup_loss, sup_acc*100, d_loss_real,unsup_racc*100, d_loss_fake,unsup_facc*100, gan_loss))
		# evaluate the model performance periodically
		if (i+1) % (bat_per_epo * 1) == 0:
			acc,filedisc,filegen= summarize_performance(i, gen_model, disc_sup, latent_dim, dataset,acc_list,n_steps)
			acc_list.append(acc*100)
	return acc_list,filedisc,filegen

#################################################################################
# TRAIN
#################################

data,n_classes,label_list=data_extract() #data extraction from npz/h5 file
dataset = load_real_samples(data,n_classes) #Define the dataset by loading real samples. (This will be a list of 2 numpy arrays, X and y)
# create the discriminator models
disc=define_discriminator(n_classes) #Bare discriminator model...
disc_sup=define_sup_discriminator(disc,lr) #Supervised discriminator model
disc_unsup=define_unsup_discriminator(disc,lr) #Unsupervised discriminator model.
gen_model = define_generator(latent_dim) #Generator
gan_model = define_gan(gen_model, disc_unsup,lr) #GAN
# train the model
# NOTE: 1 epoch = 28989/n_batch steps in this example.
acc,filedisc,filegen= train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs, n_batch)

import matplotlib.pyplot as plt 
#Discriminator accuracy plot
acc.pop(0)
plt.plot(acc)
plt.ylabel('Accuracy (%)')
plt.xlabel('Steps')
plt.savefig(f"{location}Accuracy chart.png")
plt.close()
print(f"Accuracy: {acc}")
print(f"length: {len(acc)}")
print("Accuracy Image saved")


#############################################################################
#EVALUATE THE SUPERVISED DISCRIMINATOR ON TEST DATA
# This is the model we want as a classifier.
##################################################################
from keras.models import load_model

# load the model
#n_steps=2601
disc_sup_trained_model = load_model(filedisc) #discsup{n_steps}

# load the dataset
(_, _), (testX, testy) = data  #load_data() #=====================================Added by Ankit

# expand to 3d, e.g. add channels
testX = expand_dims(testX, axis=-1)

# convert from ints to floats
testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
testX = ((testX*-1) - 127.5) / 127.5

# evaluate the model
_, test_acc = disc_sup_trained_model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))

# Predicting the Test set results
y_pred_test = disc_sup_trained_model.predict(testX)
prediction_test = np.argmax(y_pred_test, axis=1)

f=open(f"{location}dsc.txt","w")
f.write(f"Run: \n latent_dim = {latent_dim} \n n_samples ={n_samples} #multiple of 18 \n lr={lr} \n n_epochs={n_epochs} \n n_batch={n_batch}\nDisc. accuracy during training array:{acc} \nDisc. accuracy on test dataset: {test_acc * 100}\n labels: {label_list}")
f.close()
# Confusion Matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
#import scikitplot as skplt

print("Making Confusion Matrix==========")
#skplt.metrics.plot_confusion_matrix(testy, prediction_test,figsize=(200,200))
#plt.savefig("runs\\confusion matrix.png")
#plt.close()

cm = confusion_matrix(testy, prediction_test)

np.savez(f"{location}confusion_matrix.npz",array=cm)
plt.figure(figsize=(15, 15))

sns.heatmap(cm, annot=True, fmt=".1f",xticklabels=n_classes,yticklabels=n_classes)
plt.ylabel("TRUE Value")
plt.xlabel("PRED Value")
plt.savefig(f"{location}confusion matrix.png")
plt.close()
"""
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig(f"{location}confusion matrix.png")
plt.close()

#plt.savefig(f"{location}confusion matrix.png")
#print("confusion matrix saved")
#fig, ax = plt.subplots(figsize=(10, 10))
#plot_confusion_matrix(cm, testy, prediction_test, ax=ax)

from pretty_confusion_matrix import pp_matrix_from_data

plt.figure(figsize=(30, 30))
pp_matrix_from_data(testy, prediction_test,cmap='spring')
plt.savefig(f"{location}pretty_confusion_matrix.png")
plt.close()
"""
#############################################################################
#PREDICT / GENERATE IMAGES using the generator, just for fun.
##################################################################

# Plot generated images
def show_plot(examples, n):
	fig = plt.figure(figsize=(50, 50))
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :], cmap='gray_r')
	#fig.show()
	fig.savefig(f"{location}generated sample image.png")
	plt.close()
"""
		fig.subplot(n, n, 1 + i)
		fig.axis('off')
		fig.imshow(examples[i, :, :, :], cmap='gray_r')
	fig.show()    
"""

# load model
gen_trained_model = load_model(filegen) #Model trained for 100 epochs

print("Making Generator Images")
# generate images
latent_points = generate_latent_points(latent_dim, 25)  #Latent dim and n_samples

# generate images
X = gen_trained_model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0

X = (X*255).astype(np.uint8)

# plot the result
show_plot(X, 4)

print("Generator Images saved")

#import numpy as np
#print(np.__version__)
#import keras 
#print(keras.__version__)
#import tensorflow as tf
#print(tf.__version__)