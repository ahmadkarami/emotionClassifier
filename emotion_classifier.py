from keras import layers
from keras import models
from fastai.vision import *
from keras import optimizers
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("train/",target_size=(150, 150),batch_size=20,class_mode='categorical')
validation_generator = test_datagen.flow_from_directory("test/",target_size=(150, 150),batch_size=20,class_mode='categorical')

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',optimizer= "adam",metrics=['acc'])
model.fit(train_generator,steps_per_epoch=15,epochs=100,validation_data=validation_generator,validation_steps=10)
model.save('emotionRecognitionClassifier.h5')

eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory("test/",target_size=(150, 150),batch_size=20,class_mode='categorical')
eval_generator.reset()    
pred = model.predict_generator(eval_generator,5,verbose=1)

STEP_SIZE_VALID = eval_generator.n//eval_generator.batch_size
p = model.evaluate_generator(generator=eval_generator, steps=STEP_SIZE_VALID)

print("Predictions is finished")
print("loss => " ,p[0])
print("accuracy => " ,p[1])









