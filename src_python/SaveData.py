import numpy as np
from matplotlib import pyplot
import nrrd 

root="../Data/output/"

inputname = root + "input_1_1"
outputname = root + "output_1_1"

# load npy file
input_1 = np.load(inputname + ".npy")
output_1 = np.load(outputname + ".npy")

# write to a nrrd file
nrrd.write(inputname + ".nrrd", input_1)
nrrd.write(outputname + ".nrrd", output_1)


def removeChannel(vect, size_of_image):
	new = vect.reshape(size_of_image,size_of_image).astype(np.float)
	return new

def display(train_array):
	print "train array shape :", train_array.shape
	pyplot.figure(dpi=300)
	print('1')
	pyplot.set_cmap(pyplot.gray())
	print('2')
	pyplot.pcolormesh(np.flipud(train_array[:,:]))
	print('3')
	pyplot.show()


#display(removeChannel(tensor, 64))

