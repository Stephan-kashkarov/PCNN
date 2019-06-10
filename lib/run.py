import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pcnn import PCNN

random_filename = np.random.choice([
    x for x in os.listdir('data')
])

image = cv2.imread("data/"+random_filename, 0)
image = cv2.resize(image, dsize=(150, 100))
# plt.imshow(image)
# plt.show()


model = PCNN(input_shape=(100, 150), neuron_params={
    # 'yf': np.float16(0.2),
    'af': np.float16(0.6),
    # 'vf': np.float16(0.01),
    # 'al': np.float16(0.3),
    # 'vl': np.float16(0.1),
    'at': np.float16(0.6),
    'vt': np.float16(1000),
    # 'bias': np.float16(0.2),
})

history = list(model(image, iterations=16))



# a = np.zeros((100, 150))
# for x in history:
#     a = np.add(a, x)

# plt.imshow(a)
# plt.show()

history.append(image)
fig, axes = plt.subplots(4, 4)
for ax in axes.flatten():
    ax.imshow(history.pop())
plt.show()
fig, axes = plt.subplots(5, 5)
for neuron, ax in zip(model.neurons.flatten()[::600], axes.flatten()):
    his = model.history[neuron.j, neuron.i]
    for key in his.keys():
        ax.plot(his[key])

plt.show()
