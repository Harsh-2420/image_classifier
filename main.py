import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle
from tqdm import tqdm

img_size = 100
dataset = "/Users/harshjhunjhunwala/Desktop/projects/image_classifier/cat_dog_dataset/PetImages"
classes = ['Dog', 'Cat']

training_data = []
def create_train():
    for t_class in classes:
        path = os.path.join(dataset, t_class)
        class_num = classes.index(t_class)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_train()
random.shuffle(training_data)

x = []
y = []
for features, labels in training_data:
    x.append(features)
    y.append(labels)

x = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

