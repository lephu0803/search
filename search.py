import glob
import hdidx
import pickle
import keras
from keras.models import model_from_json
import keras.backend as K 
import cv2
import numpy as np
import matplotlib.pyplot as plt  
i = 0

data_embedded = 'embedded.pkl'
idx = 'Indexer_8.pkl'
image_path = sorted(glob.glob('/Users/ngocphu/Documents/Deep_Fashion/search_algorithm/output_sorted/*/*/*/*/*.jpg'))
IMAGE_SIZE = 128
# print(index_paths)
class Find_Image():
    def __init__(self):
        self.load_model()
        self.load_embedded_data()
        self.load_indexer()

    def __prewhiten(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)
        self.std_adj = np.maximum(self.std, 1.0/np.sqrt(x.size))
        self.y = np.multiply(np.subtract(x, self.mean), 1/self.std_adj)
        return self.y

    def __read_image(self, path):
        self.img = path
        self.img = cv2.resize(self.img, (IMAGE_SIZE, IMAGE_SIZE))
        self.img = self.img/255.0
        self.img = self.__prewhiten(self.img)
        return self.img

    def get_querry_embedding(self, query_path):
        self.query_img = self.__read_image(query_path)
        self.query_img1 = self.query_img.reshape((-1, self.query_img.shape[0], self.query_img.shape[1], 3))
        self.query_embedded = self.get_output([self.query_img1])
        self.query_embedded = np.array(self.query_embedded).reshape((1,1024))
        return self.query_embedded

    def load_model(self):
        with open('model.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights('Catchy_train-weights_1-6hdf5')
        self.input_1 = self.model.input
        self.embedded_output = self.model.get_layer('embedded_layer').output
        self.get_output = K.function(inputs = [self.input_1], outputs = [self.embedded_output])
    

    def load_embedded_data(self):
        with open(data_embedded, 'rb') as file:
            self.database = np.array(pickle.load(file))
        # print(database.shape)
        self.database = self.database.reshape((52712,1024))
    
    def load_indexer(self):
        with open(idx, 'rb') as fi:
            self.indexer = pickle.load(fi)
    
    def search(self, querry_image):
        self.query_embedding = self.get_querry_embedding(querry_image)
        self.ids, self.dis = self.indexer.search(self.query_embedding, 3)

        return self.ids, self.dis






    