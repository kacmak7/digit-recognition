import tensorflow as tf
import cv2
import numpy as np
import os
import neural_net

#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)

model = neural_net.neural_net()
#model.load_weights('model.h5')

imgs = os.listdir()
print(imgs)

#im = cv2.imread
