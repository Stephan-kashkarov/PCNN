import cv2
import numpy as np
import matplotlib.pyplot as plt

from pcnn import PCNN

image = cv2.imread('data/ZenH_02661_f_25_o_nf_nc_no_2015_1_e0_nl_o.jpg', 0)
image = cv2.resize(image, dsize=(150, 100))


model = PCNN(input_shape=(100, 150))

history = list(model(image, iterations=6))

history.insert(0, image)

for index, sub in enumerate(plt.subplots(1, 7)):
    sub.imshow(history[index])

plt.show()
    
