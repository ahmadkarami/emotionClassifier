from keras import layers
from keras import models
from fastai.vision import *
from keras import optimizers
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from data_preparation import preparation

preparation()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#flow_from_directory() read the images from a big folders containing images.
train_generator = train_datagen.flow_from_directory("train/",target_size=(150, 150),batch_size=20, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory("validation/",target_size=(150, 150),batch_size=20, class_mode='categorical')

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

print(model.summary())

#model.compile(loss='kullback_leibler_divergence', optimizer= "adam", metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer= "nadam", metrics=['acc'])

STEP_TRAIN = train_generator.n // train_generator.batch_size
STEP_VAL = validation_generator.n // validation_generator.batch_size

model.fit(train_generator,steps_per_epoch= STEP_TRAIN, epochs = 20
          ,validation_data = validation_generator, validation_steps= STEP_VAL)
model.save('emotionRecognitionClassifier.h5')

eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory("test/",target_size=(150, 150),batch_size=20, class_mode='categorical')
eval_generator.reset()    
pred = model.predict(eval_generator,5,verbose=1)

STEP_SIZE_VALID = eval_generator.n // eval_generator.batch_size

p = model.evaluate_generator(generator=eval_generator, steps=STEP_SIZE_VALID)

print("Predictions is finished")
print("loss => " ,p[0])
print("accuracy => " ,p[1])


