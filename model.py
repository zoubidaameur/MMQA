from tensorflow.keras.layers import MaxPooling2D ,Dense ,Dropout, Flatten
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.densenet import DenseNet169





def build_model(input_shape = (224,224,3), include_top = False, num_towers =2): 
    
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = True
    x =base_model.layers[-1].output
    x= MaxPooling2D(pool_size=(2,2))(x)
    x =Flatten()(x)
    x = Dropout(0.5)(x)

    outputs = []
    for i in range(num_towers):
        x1 = Dense(512, activation='relu')(x)
        x1 = Dropout(0.25)(x1)
        x1 = Dense(512, activation='relu')(x1)
        x1 = Dropout(0.25)(x1)
        output = Dense(1)(x1)
        outputs.append(output)  
    model = Model(inputs=base_model.layers[0].output, outputs=outputs)

    return model

