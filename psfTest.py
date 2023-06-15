import cv2
import numpy as np
from skimage.io import imread, imsave
from pyclesperanto_prototype import imshow
import pyclesperanto_prototype as cle
import pandas as pd
import matplotlib.pyplot as plt

testImage = cv2.imread('/Users/badalchowdhary/PycharmProjects/DFD_ESI/venv/IMages/1.jpg')
cropImg = testImage[1050:1350, 1450:1780]
# bead_image = imread('/Users/badalchowdhary/PycharmProjects/DFD_ESI/venv/IMages/1.jpg')
print(cropImg.shape)

# imshow(cle.maximum_x_projection(cropImg), colorbar=True)
# imshow(cle.maximum_y_projection(cropImg), colorbar=True)
# imshow(cle.maximum_z_projection(cropImg), colorbar=True)

# Segment objects
label_image = cle.voronoi_otsu_labeling(cropImg)
# imshow(label_image, labels=True)

# determine center of mass for each object
stats = cle.statistics_of_labelled_pixels(cropImg, label_image)

df = pd.DataFrame(stats)
df[["mass_center_x", "mass_center_y", "mass_center_z"]]

# configure size of future PSF image
psf_radius = 20
size = psf_radius * 2 + 1

# initialize PSF
single_psf_image = cle.create([size, size, size])
avg_psf_image = cle.create([size, size, size])

num_psfs = len(df)
for index, row in df.iterrows():
    x = row["mass_center_x"]
    y = row["mass_center_y"]
    z = row["mass_center_z"]

    print("Bead", index, "at position", x, y, z)

    # move PSF in right position in a smaller image
    cle.translate(cropImg, single_psf_image,
                  translate_x=-x + psf_radius,
                  translate_y=-y + psf_radius,
                  translate_z=-z + psf_radius)

    # visualize
    fig, axs = plt.subplots(1, 3)
    imshow(cle.maximum_x_projection(single_psf_image), plot=axs[0])
    imshow(cle.maximum_y_projection(single_psf_image), plot=axs[1])
    imshow(cle.maximum_z_projection(single_psf_image), plot=axs[2])

    # average
    avg_psf_image = avg_psf_image + single_psf_image / num_psfs

fig, axs = plt.subplots(1,3)
imshow(cle.maximum_x_projection(avg_psf_image), plot=axs[0])
imshow(cle.maximum_y_projection(avg_psf_image), plot=axs[1])
imshow(cle.maximum_z_projection(avg_psf_image), plot=axs[2])

print(avg_psf_image.min(), avg_psf_image.max())
# cv2.imshow('Image1', cropImg)
# cv2.waitKey(0)