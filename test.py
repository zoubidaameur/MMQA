#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:27:09 2020

@author: zameur
"""



from data import *
from model import *
import pickle
import pandas
import os
import datetime
from tensorflow.keras.models import load_model
import csv 
import numpy as np



def test_model(save_csv=True, db='LIVEMD', wieghts=''):

    if(db=='CIDIQ'):
        patches= 4
        test_generator = Generator(part='test', batch_size, (224,224, 3), True, 300, patches)
    elif(db=='VDID'):
        patches= 4
        test_generator = Generator(part='test', batch_size, (224,224, 3), True, 300, patches)
    elif(db=='LIVE'):
        patches= 4
        test_generator = Generator(part='test', batch_size, (224,224, 3), True, 300, patches)


    model = build_model(input_shape = (224,224,3), include_top = False, num_towers =2)
    model.load_weights('weights.h5')

    prediction1, prediction2 = model.predict_generator(generator=test_generator)

    list_labels = []
    for i in range(len(labels_path)):
        pickle_in = open(labels_path[i],'rb')
        labels = pickle.load(pickle_in)
        list_labels.append(labels)
        pickle_in.close()
    true1 = list_labels[0]
    true2 = list_labels[1]


    with open('results.csv', 'w') as f:
        fnames = ['name','prediction_distance1', 'truth_distance1', 'prediction_distance2', 'truth_distance2']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        for im in partition[part]:

            true1.append(labels50[im])
            true2.append(labels100[im])
            truEname.append(im)
            pred1=0
            pred2 = 0
            for k in range(patches):
                pred1= prediction1[(i*patches)+k]+pred1
                pred2= prediction2[(i*patches)+k]+pred2
            

        writer.writerow({'name': im,'prediction_distance1' : pred1/patches , 'truth_distance1': true1[im] ,'prediction_distance2' : pred2/patches , 'truth_distance2': true2[im]})
                                                                               
    return True

    