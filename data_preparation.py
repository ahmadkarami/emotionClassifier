import os
import random
import librosa
import numpy as np
import shutil as sh
import librosa.display
import matplotlib as plt
import scipy.io.wavfile

TEST_DIR = "test"
TRAIN_DIR = "train"
VAL_DIR = "validation"
RAVDESS_DIR = "Ravdess"

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

# ----------------------- This section will be added soon ------------------
# =============================================================================
# def split_TESS_audio(path):    
#      key_list = list(TESS_EMOTIONS.keys())
#      val_list = list(TESS_EMOTIONS.values())
#      if not os.path.exists(SPLITTED_DATASET):
#        os.mkdir(SPLITTED_DATASET)
#        for emo in EMOTIONS:
#             if not os.path.exists(SPLITTED_DATASET + "/" + EMOTIONS[emo]):
#                 os.mkdir(SPLITTED_DATASET + "/" + EMOTIONS[emo])        
#        for subdir, dirs, files in os.walk(path):
#         for file in files:
#             position = val_list.index(file.split('.')[0].split('_')[2])
#             try:
#                 src = os.path.join(subdir + "/", file)
#                 ind = key_list[position]
#                 dst = os.path.join(SPLITTED_DATASET + "/" + EMOTIONS[ind] + "/", file)
#                 sh.copyfile(src, dst)                
#             except ValueError as err:
#                 print(err)
#                 continue
# =============================================================================
# ----------------------- This section will be added soon ------------------


def split_RAVDESS_audio(path):       
       for emo in EMOTIONS:
            if not os.path.exists(TRAIN_DIR + "/" + EMOTIONS[emo]):
                os.mkdir(TRAIN_DIR + "/" + EMOTIONS[emo])        
       for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                src = os.path.join(subdir + "/", file)
                ind = int(file.split('.')[0].split('-')[2])
                dst = os.path.join(TRAIN_DIR + "/" + EMOTIONS[ind] + "/", file)
                sh.copyfile(src, dst)                
            except ValueError as err:
                print(err)
                continue

def data_augmentation(path, noise= True, stretch= False, shift= True, pitch= True):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            clip, sample_rate = librosa.load(subdir+'/'+file, sr=None)            
            if "aug" not in file: 
                if noise:
                    clip, sample_rate = librosa.load(subdir+'/'+file, sr=None)
                    noise_amp = 0.035*np.random.uniform()*np.amax(clip)
                    data = clip + noise_amp*np.random.normal(size=clip.shape[0])
                    scipy.io.wavfile.write(subdir + "/aug_noise_" + file, 16000, data)                
                if stretch:
                    clip, sample_rate = librosa.load(subdir+'/'+file, sr=None)
                    data = librosa.effects.time_stretch(clip, sample_rate)
                    scipy.io.wavfile.write(subdir + "/aug_stretch_" + file, 16000, data)                    
                if shift:
                    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                    data = np.roll(clip, shift_range)
                    scipy.io.wavfile.write(subdir + "/aug_shift_" + file, 16000, data)                    
                if pitch:
                    data = librosa.effects.pitch_shift(clip, sample_rate, 0.7)
                    scipy.io.wavfile.write(subdir + "/aug_pitch_" + file, 16000, data)

def create_spectrogram(path):    
     for subdir, dirs, files in os.walk(path):           
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
                    filename  = subdir + '/' + file.split('.')[0] + '.jpg'
                    plt.pyplot.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
                    plt.pyplot.close()    
                    fig.clf()
                    plt.pyplot.close(fig)
                    plt.pyplot.close('all')
                    del filename,clip,sample_rate,fig,ax,S
                    os.remove(subdir + '/' + file)
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
            no = len(files) * 0.15
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

def split_data_val():
    if not os.path.exists(VAL_DIR):
        os.mkdir(VAL_DIR)
        for emo in EMOTIONS:
            if not os.path.exists(VAL_DIR + "/" + EMOTIONS[emo]):
                os.mkdir(VAL_DIR + "/" + EMOTIONS[emo])            
        for subdir, dirs, files in os.walk(TRAIN_DIR):            
            x = len(files)
            no = len(files) * 0.15
            if x != 0:
                random_index = random.sample(range(x),int(no))
                random_file = [files[i] for i in random_index]                
                for file in random_file:                    
                    try:                        
                        src = os.path.join(TRAIN_DIR + "/" +  subdir.split('\\')[1] + "/", file)
                        dst = os.path.join(VAL_DIR + "/" + subdir.split('\\')[1] + "/", file)
                        sh.move(src, dst)
                    except ValueError as err:
                        print(err)
                        continue

def preparation():    
       
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
        split_RAVDESS_audio(RAVDESS_DIR)
        data_augmentation(TRAIN_DIR)
        create_spectrogram(TRAIN_DIR)
        split_data_test()
        split_data_val()