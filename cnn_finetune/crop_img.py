import numpy as np
import os
from PIL import Image
import _pickle as cPickle
import csv
import sys
from keras import backend as K

def crop_center(img,cropx,cropy):
	print(img.shape)
	y,x,z = img.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)    
	return img[starty:starty+cropy,startx:startx+cropx,0:z]

def load_data():
	
	# Create numpy array to store images
	num_train_samples = 17459
	num_test_samples = 3075
	x_train = np.zeros((num_train_samples, 224, 224, 3), dtype='uint8')
	y_train = np.zeros((num_train_samples,), dtype='uint8')
	x_test = np.zeros((num_test_samples, 224, 224, 3), dtype = 'uint8')
	y_test = np.zeros((num_test_samples,), dtype='uint8')


	f_train = open('./224/train.csv', 'r')
	f_test = open('./224/val.csv','r')
	#f_train_label = open('train_label', 'w')
	#f_test_label = open('test_label', 'w')
	try:
		data_reader = csv.reader(f_train)
		test_reader = csv.reader(f_test)
	except:
		sys.stderr.write("Cannot read {0} {1}}\n".format(f_train, f_test))
		sys.exit(1)

	count_train = 0
	for row in data_reader:
		img = Image.open('./224/' + row[0])
		arr = np.array(img)

		#when has wrong size
		try:
			arr = crop_center(arr, 224,224)
		except:
			continue

		# correct size
		# Load to x_train, y_train
		x_train[count_train,:,:,:] = arr
		y_train[count_train] = row[1]
		
		# Store to file
		#f_train_label.write(row[1]+'\n')
		#np.save('train_data', arr)
		count_train += 1
		if count_train % 100 == 0:
			sys.stdout.write("Finished {0} data".format(count_train))

	count_test = 0
	for row in test_reader:
		img = Image.open('./224/' + row[0])
		arr = np.array(img)

		#when has wrong size
		try:
			arr = crop_center(arr, 224,224)
		except:
			continue

		#correct size
		# Load to x_test, y_test
		x_test[count_test,:,:,:] = arr
		y_test[count_test] = row[1]

		# Store data to files
		#f_test_label.write(row[1]+'\n')
		#np.save('val_data', arr)
		count_test += 1
		if count_test % 100 == 0:
			sys.stdout.write("Finished {0} data".format(count_test))

	y_train = np.reshape(y_train, (len(y_train),1))
	y_test = np.reshape(y_test, (len(y_test), 1))

	# Change to the sequence compatible with load_cifar10
	x_train = x_train.transpose(0,3,1,2)
	x_test = x_test.transpose(0,3,1,2)

	if K.image_data_format() == 'channels_last':
		x_train = x_train.transpose(0, 2, 3, 1)
		x_test = x_test.transpose(0, 2, 3, 1)

	f_test.close()
	f_train.close()

	return (x_train, y_train), (x_test, y_test)
	#fclose(f_train_label)
	#fclose(f_test_label)
