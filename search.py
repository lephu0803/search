import glob
import hdidx
import pickle
import keras
from keras.models import model_from_json
import keras.backend as K 
import cv2
import urllib
import numpy as np
import base64
# import matplotlib.pyplot as plt
i = 0

data_embedded = 'embedded.pkl'
idx = 'Indexer.pkl'
database = '/Users/mai/Documents/phu/search/output_sorted/*/*/*/*/*.jpg'
image_path = sorted(glob.glob(database))
IMAGE_SIZE = 128
result_dict = {}
# print(index_paths)
class Find_Image():
    def __init__(self):
        self.load_model()
        self.load_embedded_data()
        self.load_indexer()
        self.result_dict = {}
    def __prewhiten(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)
        self.std_adj = np.maximum(self.std, 1.0/np.sqrt(x.size))
        self.y = np.multiply(np.subtract(x, self.mean), 1/self.std_adj)
        return self.y

    def __read_image(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (IMAGE_SIZE, IMAGE_SIZE))
        self.img = self.img/255.0
        self.img = self.__prewhiten(self.img)
        return self.img

    def get_querry_embedding(self, query_path):
        self.query_img = self.__read_image(query_path)
        # self.query_img = self.url_to_image(query_path)
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
    
    def url_to_image(self, url):

        with urllib.request.urlopen(url) as link:
            resp = link.read()
        self.image = np.asarray(bytearray(resp))
        self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        self.image = cv2.resize(self.image, (IMAGE_SIZE, IMAGE_SIZE))
        return self.image
    
    def update_dict(self, key, value):
        self.result_dict[key] = value
    
    def search(self, querry_image):
        self.query_embedding = self.get_querry_embedding(querry_image)
        self.ids, self.dis = self.indexer.search(self.query_embedding, 3)
        return self.ids, self.dis
    
    def find(self, array):
        self.index = array[-1]
        self.data_path =[]
        for i in self.index:
            # print(i)
            self.data_path.append(image_path[i])
        return self.data_path

    def return_json(self, data_path):
        # self.data_path
        for index, path in enumerate(data_path):
            with open(path, "rb") as imageFile:
                base64_string = base64.b64encode(imageFile.read())
            # base64_string = base64.b64encode(cv2.imread(path))
            base64_string = str(base64_string)
            base64_string = base64_string[2:-1]
            self.update_dict(str(index), base64_string)
        return self.result_dict

# def main():
#     x = Find_Image()
#     ids, dis= x.search('https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/13000934/Beagle-On-White-08.jpg')
#     similar = x.find(ids)
#     print(similar)

# main()


    