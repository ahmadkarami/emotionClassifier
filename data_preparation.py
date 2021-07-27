import os
import random
import librosa
import numpy as np
import shutil as sh
import librosa.display
import matplotlib as plt

TEST_DIR = "test"
TRAIN_DIR = "train"
WAV_DATASET_DIR = "Ravdess"
SPECTROGRAM_DATASET_DIR = "spectrogram_data_set"

EMOTIONS = {
    1:'neutral', 
    2:'calm', 
    3:'happy', 
    4:'sad', 
    5:'angry', 
    6:'fearful', 
    7:'disgust', 
    8:'surprised'
}


def create_spectrogram():
    
     for subdir, dirs, files in os.walk(WAV_DATASET_DIR):
           
           for file in files:
               try:
                    plt.interactive(False)
                    clip, sample_rate = librosa.load(subdir+'/'+file, sr=None)
                    fig = plt.figure.Figure(figsize=[0.72,0.72])
                    ax = fig.add_subplot(111)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    ax.set_frame_on(False)
                    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
                    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
                    filename  = SPECTROGRAM_DATASET_DIR + '/' + file.split('.')[0] + '.jpg'
                    plt.pyplot.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
                    plt.pyplot.close()    
                    fig.clf()
                    plt.pyplot.close(fig)
                    plt.pyplot.close('all')
                    del filename,clip,sample_rate,fig,ax,S
               except ValueError as err:
                   print(err)
                   continue


def split_data_train():
    
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)


        for emo in EMOTIONS:
            if not os.path.exists(TRAIN_DIR + "/" + EMOTIONS[emo]):
                os.mkdir(TRAIN_DIR + "/" + EMOTIONS[emo])
        
        for subdir, dirs, files in os.walk(SPECTROGRAM_DATASET_DIR):
            for file in files:
                try:
                    src = os.path.join(SPECTROGRAM_DATASET_DIR + "/", file)
                    ind = int(file.split('.')[0].split('-')[2])
                    dst = os.path.join(TRAIN_DIR + "/" + EMOTIONS[ind] + "/", file)
                    sh.move(src, dst)
                    
                except ValueError as err:
                    print(err)
                    continue


def split_data_test():
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)


        for emo in EMOTIONS:
            if not os.path.exists(TEST_DIR + "/" + EMOTIONS[emo]):
                os.mkdir(TEST_DIR + "/" + EMOTIONS[emo])
        
        
        for subdir, dirs, files in os.walk(TRAIN_DIR):
            
            x = len(files)
            no = len(files) * 0.3
            if x != 0:
                random_index = random.sample(range(x),int(no))
                random_file = [files[i] for i in random_index]
                
                for file in random_file:
                    
                    try:
                        
                        src = os.path.join(TRAIN_DIR + "/" +  subdir.split('\\')[1] + "/", file)
                        dst = os.path.join(TEST_DIR + "/" + subdir.split('\\')[1] + "/", file)
                        sh.move(src, dst)
                    except ValueError as err:
                        print(err)
                        continue


if not os.path.exists(SPECTROGRAM_DATASET_DIR):
       os.mkdir(SPECTROGRAM_DATASET_DIR)
       create_spectrogram()
       split_data_train()
       split_data_test()
      
    