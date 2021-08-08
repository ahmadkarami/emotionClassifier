from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import LSTM, Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16

def cnn_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
    model.add(BatchNormalization()) # baraye kaheshe zamane train
    model.add(Activation('elu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3)) # baraye jelogiri az overfit
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3)) # baraye jelogiri az overfit
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3)) # baraye jelogiri az overfit
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3)) # baraye jelogiri az overfit
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer= "nadam", metrics=['acc'])
    
    print(model.summary())
    
    return model



def vgg16_model():
    
    vgg_model = VGG16(weights='imagenet',include_top=False, input_shape=(150, 150, 3))

    for layer in vgg_model.layers:
        layer.trainable = False
    
    x = vgg_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=vgg_model.input, outputs=x)
    
    learning_rate= 5e-5
    transfer_model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

    
    print(transfer_model.summary())
    
    return transfer_model