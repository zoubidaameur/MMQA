#!/usr/bin/env python3

import numpy as np
import tensorflow.keras
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications 
from skimage.util import view_as_windows



class generator_overlapping(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self):
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def load_pkl(self,list_IDs_path,labels_paths,part):
        pickle_in = open(list_IDs_path,'rb')
        list_IDs = pickle.load(pickle_in)[part]
        pickle_in.close()
        list_labels = []
        for i in range(len(labels_path)):   
            pickle_in = open(labels_path[i],'rb')
            labels = pickle.load(pickle_in)
            list_labels.append(labels)
            pickle_in.close()
        return  list_IDs, list_labels


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.patches*self.batch_size, self.dim))
        y=self.init_y() 
        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(self.db_path+ ID)
            img = image.img_to_array(img)
            img = applications.densenet.preprocess_input(img)
            x = view_as_windows(np.ascontiguousarray(img),(self.dim),self.overlap_stride).reshape((-1,self.dim))
            X[(i)*self.patches :(i+1)*self.patches,:,:,:]= x
            labels = []
            for j in range(len(list_labels)):
                labels.append(list_labels[j])
            y[(i)*self.patches :(i+1)*self.patches]=labels
        return X, y
           

      
  

class Generator(generator_overlapping):
    'Generates data for Keras'
    def __init__(self,batch_size=1, dim=(224,224, 3), shuffle=True, overlap_stride=300, patches=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_dim= dim
        self.db_path=''
        self.patches= patches
        self.overlap_stride=overlap_stride
        list_IDs_path=''
        labels_paths=[]
        self.list_IDs,self.lis_labels=super().load_pkl(list_IDs_path,labels_paths,"train")
        super().__init__()
        

