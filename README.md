# Speech-Emotion-Classification

I encounter this project on a Hackathon named intelligence Augmentation...
The problem statement of the project was to detect the emotions in a speech via dataset provided...



## Dataset Description:

Data set info Link : https://drive.google.com/drive/folders/1LtZleSzLv1iEHn26dd154VXAb4vVnAa2?usp=sharing

The folder contains the following files...
1) Train.csv (contains 2 columns: "filename", "emotion")
2) Test.csv (only contain filename column)
3) TrainAudioFiles (contains 5816 audio files)
4) TestAudioFiles (contains 2482 audio files)



## APPROACH


### 1)	Audio Classification (Changing the format from mp3 to wap).ipynb file:**

- The dataset provided consist of both “.mp3” and “.wav” format of files. So we converted the all files in “.wav” format for processing the data. 
    
- In this we install ***Pydub***, it is a library to deal with audio files…Use to split, merge or edit the audio files. 
   
- For splitting the name of the file with its extension and also for converting the audio files from .mp3 to .wav, we use: ***AudioSegment*** method which is a wrapper method in pydub 

- We converted the both train and test files from .mp3 extension to .wav extension.
   
- Then we save these converted folders as *train_wav*  and *test_wav*
    
    
          

### 2)	Audio Classification ( EDA ).ipynb file:

- The second file was where we visualise the data and perform Exploratory Data Analysis on the training and testing set.

- In this we install and import ***Librosa library***, It is basically used when we work with audio data like in music generation(using LSTM's), Automatic Speech Recognition. It provides the building blocks necessary to create the music information retrieval systems.
    
- This actually helps us to really work well with the sound signals, by using this we can read the sound signals, finds the sample rate, get to know about the channels...
    
- We also check the ***Sample Rate*** of the audio files.A sample rate defines how many times per second a sound is sampled.
    
- Usually the avg sample rate is 44.1 KH but here we are getting around 22.05 kH sample rate.
    
- Then in this file we discuss about why we use librosa only to perform EDA




### 3) Audio Classification ( Data Preprocessing ).ipynb file: 

- Third file was for pre-processing of data where we extracted features from the data
    
- In this we use ***Mel-Frequency-Cepstral-Coefficients (MFCC)*** method of librosa to extract the features from audio files. The MFCC summarises the frequency distribution accross the window size, so it is possible to analyse both frequency and time characterstics of sound. These audio representations allow us to identify the features for classification.
    
- Then we also define a function to extract the features **“feature_extractor”** which takes a filename, then in the function we will extract the audio data and the sample rate from librosa.load(), after that we will create features for the file using mfcc and at last finally to find out the scaled feature we will do the mean on the transpose of the particular mfcc_features that we are getting.
    
- Then we analyse the progess through ***“tqdm”***  library in python
    
- We did this for both train and test files.
    
- Save the files with extracted features as *extracted_features.csv* and *extracted_features_test.csv*
    
    
    
  
### 4) Audio Classification ( Model Training and Testing ).ipynb file:**

- The fourth and the last file is of Model Training and Testing, we perform model training of the extracted files…
   
- We first import the dataset and convert the target features into dummies by LabelEncoding. 
    
- Then we use **Pycaret** library in python to compare the classification model.
    
- We get that RandomForest has the highest accuracy and fits the best to the model.
  
- Then we used ***OPTUNA hyperparameter optimisation*** to tune our model, via this we get the accuracy od 56.5% 
   
- And then we change the n_jobs = -2 and the accuracy increased little bit which was 57.6%

- Then we predict the emotion on the test data provided and save the file as prediction.csv



