import tensorflow as tf
import cv2 as cv2
import numpy as np
from tensorflow.keras.models import load_model
from configparser import ConfigParser

read_conf = ConfigParser()
read_conf.read('../config.ini')

model = load_model('./Models/SqueezeUnet.h5')

print(model.summary())

input_img = cv2.imread(read_conf['PATHS']['path_to_input'])
ground_truth = cv2.imread(read_conf['PATHS']['path_to_ground_truth'])
ExG = cv2.imread(read_conf['PATHS']['path_to_exG_output'])

input_img = np.expand_dims(input_img, axis=0)

start = cv2.getTickCount()

predict = model.predict(input_img, verbose=1)
predict[predict > 0.8] = 255
predict[predict <= 0.8] = 0

stop = cv2.getTickCount()
print(str((stop - start)/cv2.getTickFrequency()*1000) + " ms")

cv2.imwrite(read_conf['PATHS']['path_to_squeezeunet_output'], np.squeeze(predict))

cv2.imshow("Squeeze Unet output", np.squeeze(predict))
cv2.imshow("Ground truth image", ground_truth)
cv2.imshow("ExG + Otsu output", ExG)
cv2.imshow("Original image", np.squeeze(input_img))

cv2.waitKey(30000)  

cv2.destroyAllWindows() 