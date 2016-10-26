import nrrd 
import numpy as np

image, options = nrrd.read("stx_noscale_204948_V06_t1w_RAI_maskedOutputWarped_z80Square0.nrrd")

nrrd.write("test.nrrd", image, options)

image2, options2 = nrrd.read("test.nrrd")

print image, options

save = {
	'image1':options,
	'image2':options2,
}


table = []
table.append(options)
table.append(options2)
#table = np.ndarray(shape=(3)
#table[1] = options
#table[2] = options2

np.savez("header.npz", image1=save['image1'], image2=save['image2'])

headers = np.load("header.npz")
print("Image 1:")
print(headers["image1"])
print("Image 2")
print(headers["image2"])

