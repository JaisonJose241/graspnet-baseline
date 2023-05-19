# Python program to convert
# numpy array to image

# import required libraries
import numpy as np
from numpy import load
from PIL import Image as im

# define a main function
def main():

	# create a numpy array from scratch
	# using arange function.
	# 1024x720 = 737280 is the amount
	# of pixels.
	# np.uint8 is a data type containing
	# numbers ranging from 0 to 255
	# and no non-negative integers
	array = load("/home/jsn/graspnet-baseline/doc/example_data/demo/depth_raw_0.npy")
	
	# check type of array
	print(type(array))
	
	# our array will be of width
	# 737280 pixels That means it
	# will be a long dark line
	print(array.shape)
	
	# Reshape the array into a
	# familiar resoluition
	array = np.reshape(array, (480, 640))
	
	# show the shape of the array
	print(array.shape)

	# show the array
	print(array)
	
	# creating image object of
	# above array
	data = im.fromarray(array)
	data = data.convert('RGB')
	
	# saving the final output
	# as a PNG file
	data.save('/home/jsn/graspnet-baseline/doc/example_data/demo/depth_raw_0.png')

# driver code
if __name__ == "__main__":
	
    # function call
    main()
