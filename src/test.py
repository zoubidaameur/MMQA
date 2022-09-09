from data import *
from model import *
import pickle
import pandas
import os
from tensorflow.keras.models import load_model
import csv 
import numpy as np



def test_model(save_csv=True, db='LIVEMD', wieghts=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIDIQ' , type=str, help='CIDIQ, VDID, LIVE')
    parser.add_argument('--weight', default='weights.h5' , type=str, help='Path to weights')
    parser.add_argument('--num_patches', default=4, type=int, help='Number of patches')
    list_IDs_path = args.dataset + ".pickle"
    patches = args.patches
    list_IDs = pd.read_pickle(r'list_IDs_path')

    if(args.dataset=='CIDIQ' or args.dataset=='VDID'):
        test_generator = Generator('test', 1, (224,224, 3), False, 300, patches)
        model = build_model(input_shape = (224,224,3), include_top = False, num_towers =2)        

    elif(args.dataset=='LIVE'):
        test_generator = Generator_LIVE('test', 1, (224,224, 3), False, 300, patches)
        model = build_model(input_shape = (224,224,3), include_top = False, num_towers =7)

    model.load_weights('weights.h5')
    predictions = model.predict_generator(generator=test_generator)

    with open('results.csv', 'w') as f:
        for i in range(len(list_IDs)):
            if(args.dataset=='CIDIQ' or args.dataset=='VDID'): 
                pred1= pred2 = 0
                for k in range(patches):
                    pred1= predictions[0][(i*patches)+k]+pred1
                    pred2= predictions[1][(i*patches)+k]+pred2
                writer.writerow({'name': list_IDs[i],'prediction_distance1' : pred1/patches , 'prediction_distance2' : pred2/patches})
            elif(args.dataset=='LIVE'):
                pred1= pred2 = pred3 =pred4 = pred5 =pred6 = pred7 = 0
                for k in range(patches):
                    pred1= predictions[0][(i*patches)+k]+pred1
                    pred2= predictions[1][(i*patches)+k]+pred2
                    pred3= predictions[2][(i*patches)+k]+pred3
                    pred4= predictions[3][(i*patches)+k]+pred4
                    pred5= predictions[4][(i*patches)+k]+pred5
                    pred6= predictions[5][(i*patches)+k]+pred6
                    pred7= predictions[6][(i*patches)+k]+pred7
                writer.writerow({'name': list_IDs[i],'prediction_distance1' : pred1/patches , 'prediction_distance2' : pred2/patches, 'prediction_distance3' : pred3/patches, 'prediction_distance4' : pred4/patches, 'prediction_distance5' : pred5/patches, 'prediction_distance6' : pred6/patches, 'prediction_distance7' : pred7/patches})                                                                        
    return True

    