import sys
from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.keras.callbacks import TensorBoard
from data import *
from model import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIDIQ' , type=str, help='CIDIQ, VDID, LIVE')
    parser.add_argument('--batch_size', default= 4 ,type=int)
    parser.add_argument('--epochs',default= 30 ,type=int)
    args = parser.parse_args() 


    if(args.dataset=='CIDIQ'):
        training_generator = Generator('train', args.batch_size, (224,224, 3), True, 300, 4)
        validation_generator = Generator('test', args.batch_size, (224,224, 3), True, 300, 4)
        losses= {'mean_squared_error','mean_squared_error'}
        lossWeights ={0.5, 0.5}
    elif(args.dataset=='VDID'):
        training_generator = Generator('train', args.batch_size, (224,224, 3), True, 300,4)
        validation_generator = Generator('test', args.batch_size, (224,224, 3), True, 300, 4)
        losses= {'mean_squared_error','mean_squared_error'}
        lossWeights ={0.5, 0.5}
    elif(args.dataset=='LIVE'):
        training_generator = Generator_LIVE('train', args.batch_size, (224,224, 3), True, 300,4)
        validation_generator = Generator_LIVE('test', args.batch_size, (224,224, 3), True, 300, 4)
        losses= {'mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error'}
        lossWeights ={1,1,1,1,1,1,1}


    model = build_model(input_shape = (224,224,3), include_top = False, num_towers =2)
    model.compile(optimizer=Adam(lr=0.0001), loss= losses, loss_weights=lossWeights)
    history =model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True,workers=4,epochs=args.epochs)
    out_model_path = db+'_epochs:'+str(epochs)
    model.save_weights(out_model_path +'.h5')
    return True



