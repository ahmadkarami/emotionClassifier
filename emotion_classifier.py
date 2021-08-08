from fastai.vision import *
from keras import optimizers
from data_preparation import *
from keras.callbacks import *
from custom_models import *
from keras.preprocessing.image import ImageDataGenerator

models = []
trained_models = []
histories =[]

preparation()
models.append(cnn_model()) # ba val_acc kar mikone
models.append(vgg16_model()) # ba val_accuracy kar mikone


# ----------------------------------------------------------------------------
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

#flow_from_directory() read the images from a big folders containing images.
train_gen = train_data.flow_from_directory(TRAIN_DIR,target_size=(150, 150),batch_size=20, class_mode='categorical')
validation_gen = val_data.flow_from_directory(VAL_DIR,target_size=(150, 150),batch_size=20, class_mode='categorical')
test_gen = test_data.flow_from_directory(TEST_DIR,target_size=(150, 150),batch_size=20, class_mode='categorical')

TRAIN_STEP = train_gen.n // train_gen.batch_size
VAL_STEP = validation_gen.n // validation_gen.batch_size
TEST_STEP = test_gen.n // test_gen.batch_size
# if you forget to reset the test_gen you will get weird outputs.
test_gen.reset()
# ----------------------------------------------------------------------------

# -------------------------------------fit-------------------------------------
for i in range(len(models)):
    checkPoint_pth = 'emotionClassifierModel_'+str(i)+'.hdf5'
    if i == 0:
        VA = "val_acc"
    else:
        VA = "val_accuracy"
    earlyStopping = EarlyStopping(monitor=VA, patience=25, verbose=1, restore_best_weights=False)
    reduceLR = ReduceLROnPlateau(monitor=VA, factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    modelCheckpoint = ModelCheckpoint(checkPoint_pth, monitor = VA, verbose=1, save_best_only=True, mode='max')
    TB = TensorBoard()
    callbacks = [earlyStopping, reduceLR, modelCheckpoint, TB]
    
    history = models[i].fit(train_gen, validation_data = validation_gen,
              steps_per_epoch = TRAIN_STEP, validation_steps = VAL_STEP,
              epochs = 2, verbose = 1, shuffle = True, callbacks = callbacks)
    
    trained_models.append(models[i])
    histories.append(history)
# -------------------------------------fit-------------------------------------

# -------------------------------------predict---------------------------------
for i in range(len(models)):
    pred = models[i].predict(test_gen, 5, verbose=1)
    p = models[i].evaluate_generator(generator = test_gen, steps = TEST_STEP)
    
    print("Predictions for model ",i ," is finished")
    print("loss => " ,p[0])
    print("accuracy => " ,p[1])
# -------------------------------------predict---------------------------------


