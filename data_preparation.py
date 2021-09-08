import os
import random
import re
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
TESS_DIR = "TESS"
CREMA_DIR = "CREMA"
SAVEE_DIR = "SAVEE"

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

TESS_EMOTIONS = {
    1:'neutral',  
    3:'happy', 
    4:'sad', 
    5:'angry', 
    6:'fear', 
    7:'disgust', 
    8:'ps'
   }

CREMA_EMOTIONS = {
    1:'NEU',  
    3:'HAP', 
    4:'SAD', 
    5:'ANG', 
    6:'FEA', 
    7:'DIS'
   }

SAVEE_EMOTIONS = {
    1:'n', 
    3:'h', 
    4:'sa', 
    5:'a', 
    6:'f', 
    7:'d', 
    8:'su'
}

def split_SAVEE_audio(path):    
    
     key_list = list(SAVEE_EMOTIONS.keys())
     val_list = list(SAVEE_EMOTIONS.values())

     for subdir, dirs, files in os.walk(path):
       for file in files:
         fileName = file.split('.')[0]
         
         match = re.match(r"([a-z]+)([0-9]+)", fileName, re.I)
         if match:
            items = match.groups()
            try:
                src = os.path.join(subdir + "/", file)
                ind = key_list[val_list.index(items[0])]
                dst = os.path.join(TRAIN_DIR + "/" + EMOTIONS[ind] + "/", file.split('.')[0] + subdir.split('\\')[1] + ".wav")
                sh.copyfile(src, dst)                
            except ValueError as err:
                print(err)
                continue

def split_CREMA_audio(path):    
     key_list = list(CREMA_EMOTIONS.keys())
     val_list = list(CREMA_EMOTIONS.values())

     for subdir, dirs, files in os.walk(path):
      for file in files:
         position = val_list.index(file.split('.')[0].split('_')[2])
         try:
             src = os.path.join(subdir + "/", file)
             ind = key_list[position]
             dst = os.path.join(TRAIN_DIR + "/" + EMOTIONS[ind] + "/", file)
             sh.copyfile(src, dst)                
         except ValueError as err:
             print(err)
             continue

def split_TESS_audio(path):    
     key_list = list(TESS_EMOTIONS.keys())
     val_list = list(TESS_EMOTIONS.values())

     for subdir, dirs, files in os.walk(path):
      for file in files:
         position = val_list.index(file.split('.')[0].split('_')[2])
         try:
             src = os.path.join(subdir + "/", file)
             ind = key_list[position]
             dst = os.path.join(TRAIN_DIR + "/" + EMOTIONS[ind] + "/", file)
             sh.copyfile(src, dst)                
         except ValueError as err:
             print(err)
             continue

def split_RAVDESS_audio(path):       

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
    for subdir, dirs, files in os.walk(TRAIN_DIR):   
            print(subdir)
    if not os.path.exists(VAL_DIR):
        os.mkdir(VAL_DIR)
        for emo in EMOTIONS:
            if not os.path.exists(VAL_DIR + "/" + EMOTIONS[emo]):
                os.mkdir(VAL_DIR + "/" + EMOTIONS[emo])            
        for subdir, dirs, files in os.walk(TRAIN_DIR):   
            print(subdir)
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
        
        for emo in EMOTIONS:
          if not os.path.exists(TRAIN_DIR + "/" + EMOTIONS[emo]):
            os.mkdir(TRAIN_DIR + "/" + EMOTIONS[emo])
        
        split_SAVEE_audio(SAVEE_DIR)
        split_CREMA_audio(CREMA_DIR)
        split_RAVDESS_audio(RAVDESS_DIR)
        split_TESS_audio(TESS_DIR)
        data_augmentation(TRAIN_DIR)
        create_spectrogram(TRAIN_DIR)
        split_data_test()
        split_data_val()
