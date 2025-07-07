"""
Run the following line to install nighty build of sklearn before importing libraries (uninstall of old sklearn required):
"""

!pip uninstall scikit-learn -y
!pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn

# Import ds libraries.
import numpy as np
import pandas as pd
from sklearn.metrics import det_curve, DetCurveDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Import audio libraries.
import librosa
from librosa import display

# Import utilities.
import os
import glob
from urllib.request import urlretrieve
from pathlib import Path
import random
from zipfile import ZipFile
import ast

# Import plotting
import matplotlib.pyplot as plt

# Directory scan function.
def dirscan(dir):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(dirscan(dir))
    return subfolders

# Define pre-emphasis function.
def preemphasis(signal, coeff=0.95):

    # Emphassized signal
    signal_out = np.append(signal[0], signal[1:] - coeff * signal[:-1])

    # Convert to numpy-array.
    signal_out = np.array(signal_out)

    # Return pre-emphasized signal
    return signal_out


# Get MFCCs.
def mfccs(file_path: str,
          preemphasis_coeff: float = 0.95,
          n_fft: int = 512,
          winlen: float = 0.025,
          winstep: float = 0.01,
          n_filter_banks: int = 40,
          n_mfccs: int = 13,
          winfunc: callable = np.hamming,
          sampling_rate=8000,
          verbose: bool = False) -> np.ndarray:

    # Reassemble the sampling rate if needed.
    y, sr = librosa.load(file_path, sr=None)
    if sr != sampling_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sampling_rate)

    # Pre-emphasize the signal.
    y = preemphasis(y, preemphasis_coeff)

    # Define window length and hop-length.
    window_length = int(winlen * sampling_rate)
    hop_length = int(winstep * sampling_rate)

    # Get magnitude spectrogram.
    spectrogram = np.abs(librosa.stft(y,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      win_length=window_length,
                                      center=False))
    # Get mel-spectrogram.
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram)

    # Get MFCC features from the mel-spectrogram..
    mfccs = librosa.feature.mfcc(S=np.log(mel_spectrogram), n_mfcc=n_mfccs)

    # Normalize MFCCs by mean and standard-deviation.
    mfccs_norm = mfccs - np.mean(mfccs, axis=1, keepdims=True)
    mfccs_norm /= np.std(mfccs, ddof=0, axis=1, keepdims=True)

    # Visualization for debugging
    if verbose:
        plt.figure(figsize=(10, 12))
        librosa.display.waveplot(y, sr=sr, ax=plt.subplot(3, 2, 1))
        plt.gca().set_title('Wave plot')
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram),
                                 sr=sr,
                                 hop_length=hop_length,
                                 fmax=sampling_rate / 2,
                                 y_axis='linear',
                                 x_axis='time',
                                 ax=plt.subplot(3, 2, 2))
        plt.gca().set_title('Spectrogram')
        librosa.display.specshow(librosa.amplitude_to_db(mel_spectrogram),
                                 sr=sr,
                                 hop_length=hop_length,
                                 fmax=sampling_rate / 2,
                                 y_axis='mel',
                                 x_axis='time',
                                 ax=plt.subplot(3, 2, 3))
        plt.gca().set_title('Mels-Spectrogram')
        librosa.display.specshow(librosa.amplitude_to_db(mfccs),
                                 sr=sr,
                                 fmax=sampling_rate / 2,
                                 x_axis='time',
                                 hop_length=hop_length,
                                 ax=plt.subplot(3, 2, 4))
        plt.gca().set_title('MFCCs')
        ax = plt.subplot(3, 2, 5)
        ax.plot(mfccs)
        ax.set_title('Individual MFCC features')
        ax = plt.subplot(3, 2, 6)
        ax.plot(mfccs_norm)
        ax.set_title('Normalized MFCCs')
        plt.tight_layout()
    return mfccs_norm.T

# Function for unifying MFCC frame-length.
def unify_mfcc(standard_length, mfcc):

    # If maximum length > frame length, do padding to end.
    if (standard_length > mfcc.shape[0]):
        pad_width = standard_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0,pad_width), (0,0)), mode='constant', constant_values=(0,0))

    # Else do cutoff from end.
    else:
        mfcc = mfcc[:standard_length, :]

    # Return unified MFCC.
    return mfcc

"""Download the data set and get utterances:"""

# Download data set and extract files.
url = r"https://github.com/soerenab/AudioMNIST/archive/master.zip"
filename = "/tmp/data.zip"
if not os.path.exists(filename):
    print('Downloading AudioMNIST dataset ...')
    urlretrieve(url, filename=filename)
path = '/tmp/AudioMNIST-master/data'
if not os.path.exists(path):
    print("Extracting files from AudioMNIST...")
    with ZipFile(filename, mode='r') as f:
        f.extractall(path="/tmp")

# Get utterances.
path = "/tmp/AudioMNIST-master/data"
subfolders = dirscan(path)

utterances = []

# Iterate through subfolders and add all wavfiles under same variable.
for folder in subfolders:
    path = Path(folder).absolute()
    for wavefile in glob.glob(os.path.join(path, '*.wav')):
        utterances.append(wavefile)

np.random.shuffle(utterances)

metadata = [os.path.basename(i).split('_')[:-1] for i in utterances]
digits = np.array([i[0] for i in metadata])
speakers = np.array([i[1] for i in metadata])

print(utterances[:5])
print(metadata[:5])
print(digits[:5])
print(speakers[:5])

"""You can mount your Google Drive, if you want to save following pickle/csv-files etc to you Drive:"""

# Mount Google Drive.
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

"""Create set of unique speakers for later use:"""

# Unique speakers.
speakers_set = np.unique(speakers)

"""Following code block is the most time-requiring among data processing as it loads the whole mfcc-features set and also creates the "main" DataFrame from the utterances and it's metadata (depending on machine, usually took ~20min with GPU-acceleration and using NVidia GTX1050 on local machine and ~1h with Colab). After run once, this block can be skipped by loading the saved Pickle-file. (if needed, you can also load it to you Google Drive in order to not be distracted by the runtime-disconnection issues)"""

# Split to Train Sets and Test Sets based on following conditions:

# DataFrame containing all utterances, speakers and digits.
all_data = pd.DataFrame()

# Counter variable.
counter = 0

# Define standard length for frames in MFCC.
standard_length = 60

# For utterance.
for file in utterances:
    # Do processing into utterance, digit and speaker.
    splitted = os.path.basename(file).split('_')[:-1]
    digit = int(splitted[0])
    speaker = int(splitted[1])  # Get rid of zeros for easier indexing.
    feats = mfccs(file) # Get mfcc.
    #Reshape to rough estimate in order to maintain unified shape between samples.
    feats = unify_mfcc(standard_length, feats)
    # Just in case save also to its own 3D array.
    all_data = all_data.append(pd.DataFrame({"utt": ["{}".format(file)], "dgt": [digit], "spk": [speaker], "mfcc": [feats]}, index=[counter]))
    counter += 1

# Save to Pickle for later use to runtime.
all_data.to_pickle("/tmp/meta_mfcc.pkl")

"""Run the following block if you want to save Pickle to your Drive:"""

# Save Pickle to Google Drive. Path can be changed if wanted to specify different folder.
path = "/content/drive/MyDrive/meta_mfcc.pkl"
all_data.to_pickle("{}".format(path))

"""Run the following block if you want to load the Pickle from runtime:"""

# Load Pickle from runtime (if still there)
all_data = pd.read_pickle("/tmp/meta_mfcc.pkl")
print(all_data.head())

"""Run the following if you want to load Pickle from the Google Drive:"""

# Load Pickle from Google Drive.
path = "/content/drive/MyDrive/meta_mfcc.pkl"
all_data = pd.read_pickle("{}".format(path))
print(all_data.head())

"""Split data into 4 different Train Sets and Test Sets:"""

"""Train Set and Test Set 1: Contains "basic" scenario, where all data is splitted into Train Set and Test Set, later on the train set 
is used to train the network by picking random samples from the Train Set and is then tested using "target scores" containing 10 randomly 
chosen utterances from the same speaker and 50 randomly chosen utterances from an other speaker on the Test Set. Please notice, that the
target scores and non-target scores are generated from the Test Set in the evaluation phase and only the main Set for testing is generated 
here."""

# Train Set 1.
train_set_basic = pd.DataFrame()

# Test Set 1.
test_set_basic = pd.DataFrame()

# New speakers set (formatted).
speakers_set_formatted = []

# Process digits away from unique speakers set.
for speaker in speakers_set:
    speaker = int(str(speaker))
    speakers_set_formatted.append(speaker)

# Take 40 speakers at random from the  spekers_set as Training Set.
train_spk = []
sample_speakers = random.sample(list(speakers_set_formatted), len(speakers_set_formatted) - 20)
for speaker in sample_speakers:
    train_spk.append(speaker)

# Take remaining 20 speakers (not in Train Set) as Test Set.
test_spk = []
for speaker in speakers_set_formatted:
    if np.isin(speaker, train_spk) == False:
        test_spk.append(speaker)

# For each speaker in Train Set speakers.
for i in range(len(train_spk)):
    wanted = train_spk[i]
    speaker = all_data["spk"] == wanted
    train_set_basic = train_set_basic.append(all_data[speaker])

# For each speaker in Test Set speakers.
for i in range(len(test_spk)):
    wanted = test_spk[i]
    speaker = all_data["spk"] == wanted
    test_set_basic = test_set_basic.append(all_data[speaker])

print("Train Set Basic head:\n {}\nShape: {}".format(train_set_basic.head(), train_set_basic.shape))
print("Test Set Basic head:\n {}\nShape: {}".format(test_set_basic.head(), test_set_basic.shape))


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(train_set_basic["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Train Set Basic speakers')
plt.subplot(1, 2, 2)
plt.hist(test_set_basic["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Test Set Basic speakers')
plt.tight_layout()
plt.show()

"""Train Set and Test Set 2: Contains context Dependent data (only same digit)
   Data is splitted so, that the Train Set and Test set both contain only same digits from all speakers
   for example digit 1 from all speakers. Train Set contains 50% of the digits and the Test set other 50%.
   Note that the 50% mark is not necessarily the splitting distribution between the digits spoken from same speaker 
   (set may contain 100% of digits y from speaker x and 0% from an other in this scenario)."""

# Variable to define which digit is used for set making.
used_digit = 4

# DF for all utterances from chosen digit.
all_utts_from_digit = pd.DataFrame()

# Get all utts based on digit.
wanted = all_data["dgt"] == used_digit
all_utts_from_digit = all_data[wanted]

# Train Set 2.
train_set_dependent = pd.DataFrame()

# Test Set 2.
test_set_dependent = pd.DataFrame()

# Split into half, other half as Train Set and other as Test set.
train_set_dependent, test_set_dependent = train_test_split(all_utts_from_digit, test_size=0.5)

print("Train Set context Dependent head:\n {}\nShape: {}".format(train_set_dependent.head(), train_set_dependent.shape))
print("Test Set context Dependent head:\n {}\nShape: {}".format(test_set_dependent.head(), test_set_dependent.shape))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(train_set_dependent["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Train Set Dependent speakers')
plt.subplot(1, 2, 2)
plt.hist(test_set_dependent["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Test Set Dependent speakers')
plt.tight_layout()
plt.show()

"""Train Set and Test Set 3: Contains context Limited data, where the whole data set is included.
   With this setup, 50% of the digits and utterances per speaker are distributed into Train Set and the other 50% into Test Set."""

# Train Set 3.
train_set_limited = pd.DataFrame()

# Test Set 3.
test_set_limited = pd.DataFrame()

# For number of digits.
for j in range(0, 10):
    # For number of speakers.
    for i in range(1, 61):

        # Get speaker and digit according to loop iterators.
        speakers = all_data["spk"] == i
        digits = all_data["dgt"] == j

        # Gather data cells from DF, where digit and speaker are
        # corresponding to the loop iterators.
        data = all_data[speakers & digits]

        # Divider for the halves to be added to sets.
        train_temp, test_temp = train_test_split(data, test_size=0.5)

        # Append halves to actual sets.
        train_set_limited = train_set_limited.append(train_temp)
        test_set_limited = test_set_limited.append(test_temp)

print("Train Set Text Limited head:\n {}\nShape: {}".format(train_set_limited.head(), train_set_limited.shape))
print("Test Set Text Limited head:\n {}\nShape: {}".format(test_set_limited.head(), test_set_limited.shape))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(train_set_limited["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Train Set Limited speakers')
plt.subplot(1, 2, 2)
plt.hist(test_set_limited["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Test Set Limited speakers')
plt.tight_layout()
plt.show()

"""Train Set and Test Set 4: Contains context Independent data, where Train Set contains all utterances of 5 digits from the Data Set
   and the Test Set contains an other 5 digits from the Data Set (which are not in the Train Set)"""

# Generate list of digits from 0-9.
generated_digits = []
for i in range(10):
    generated_digits.append(i)

# Take digits for Train Set from generated list using random sampling.
train_digits = random.sample(list(generated_digits), len(generated_digits) // 2)

# Populate Test Set digits from the ones which were not included into Train Set digits.
test_digits = []
for digit in generated_digits:
    if np.isin(digit, train_digits) == False:
        test_digits.append(digit)

print("Train Set digits: {}".format(train_digits))
print("Test Set digits: {}".format(test_digits))

# Train Set 4.
train_set_independent = pd.DataFrame()

# Test Set 4.
test_set_independent = pd.DataFrame()

for i in range(5):
    wanted = train_digits[i]
    wanted_digit = all_data["dgt"] == wanted
    train_set_independent = train_set_independent.append(all_data[wanted_digit])
for j in range(5):
    wanted = test_digits[j]
    wanted_digit = all_data["dgt"] == wanted
    test_set_independent = test_set_independent.append(all_data[wanted_digit])

print("Train Set context Independent head:\n {}\nShape: {}".format(train_set_independent.head(), train_set_independent.shape))
print("Test Set context Independent head:\n {}\nShape: {}".format(test_set_independent.head(), test_set_independent.shape))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(train_set_independent["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Train Set Independent speakers')
plt.subplot(1, 2, 2)
plt.hist(test_set_independent["spk"])
plt.xlabel("Speaker ID")
plt.ylabel("Samples")
plt.title('Test Set Independent speakers')
plt.tight_layout()
plt.show()

"""You can test MFCC-fetching with this premade function (fetches MFCCs of one utterance per Set and visualized the process):"""

# Testing MFCC-fetch once for each set to see they work.
def test_mfcc_fetch():
    # Set verbose to True, if you want to plot the results.
    
    # Context Basic Sets.
    _ = mfccs(train_set_basic["utt"].iloc[0], verbose=True)
    _ = mfccs(test_set_basic["utt"].iloc[0], verbose=True)

    # Context Dependent Sets.
    _ = mfccs(train_set_dependent['utt'].iloc[0], verbose=True)
    _ = mfccs(test_set_dependent['utt'].iloc[0], verbose=True)

    # Context Limited Sets.
    _ = mfccs(train_set_limited['utt'].iloc[0], verbose=True)
    _ = mfccs(test_set_limited['utt'].iloc[0], verbose=True)

    # Context Independent Sets.
    _ = mfccs(train_set_independent['utt'].iloc[0], verbose=True)
    _ = mfccs(test_set_independent['utt'].iloc[0], verbose=True)

test_mfcc_fetch()

# Build Base Model for Siamese Neural Network.
def build_model():
    # Create base model.
    base_model = keras.Sequential([

        # 2 times convolutions and max pooling.
        keras.layers.Conv2D(name="1_C", filters=32, kernel_size=(5,5), padding="same", input_shape=(60, 13, 1), activation ="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(name="M_1", pool_size=2, strides=2),

        keras.layers.Conv2D(name="2_C", filters=40, kernel_size=(5,5), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(name="M_2", pool_size=2, strides=2),

        # 2 times convolutions without max pooling.
        keras.layers.Conv2D(name="3_C", filters=40, kernel_size=(1,1), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(name="4_C", filters=12, kernel_size=(3,3), padding="same"),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(432, activation="relu"),
        keras.layers.Dense(60, activation="relu"),
    ])

    # Initialize optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Adam  optimizer.

    base_model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=["accuracy"],
                       sample_weight_mode=None)

    base_model.build((60, 13, 1))

    print("Base Model initialized successfully")

    return base_model

"""Build Siamese Model:"""

# Build a Siamese Neural Network for Automatic Speaker Verification (AVS).
def build_siamese(base_model, input_shape):

    # Set inputs to shape.
    sample1 = tf.keras.Input(shape=input_shape)
    sample2 = tf.keras.Input(shape=input_shape)

    # Get feature maps by samples from NNW model.
    encoded_sample1 = base_model(sample1)
    encoded_sample2 = base_model(sample2)

    # Calculate euclidean distance.
    euclidean_dist = Subtract()([encoded_sample1, encoded_sample2])
    euclidean_dist = Lambda(lambda x: K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True)))(euclidean_dist)

    # Output single sigmoidian number defining if two sample utterances were from same speaker (~1 = same, ~0 = different).
    prediction = keras.layers.Dense(1, activation="sigmoid")(euclidean_dist)

    siamese_model = Model(inputs=[sample1, sample2], outputs=prediction)

    # Initialize optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer.

    siamese_model.compile(optimizer = optimizer, loss = keras.losses.binary_crossentropy, metrics = ["accuracy"], sample_weight_mode=None)

    siamese_model.build((120, 26, 1))

    print("Siamese model build succesfully")

    return siamese_model

"""Function to fit the Model:"""

# Trains the given Siamese Model with given Train Set.
def train_siamese(train_samples: pd.DataFrame, siamese_model: Model = None, verbose: bool = False):

    # Get indices from Sample Set.
    indices = np.arange(len(train_samples))
    np.random.shuffle(indices)
    indices_clone = np.copy(indices)
    np.random.shuffle(indices_clone)

    # Generate Sample Set as pairwise DataFrame based on indices.
    pairwise_map = pd.DataFrame({"sample1": indices, "sample2": indices_clone})

    # Take speakers from Sample Set.
    speaker_sample1 = train_samples["spk"].iloc[pairwise_map["sample1"].values].values
    speaker_sample2 = train_samples["spk"].iloc[pairwise_map["sample2"].values].values

    # Take MFCCs for Sample Set.
    mfccs_1 = train_samples["mfcc"].iloc[speaker_sample1].values
    mfccs_2 = train_samples["mfcc"].iloc[speaker_sample2].values

    # Generate correct input format for NNW.
    mfccs_inputs_1 = np.zeros((len(train_samples), 60, 13))
    mfccs_inputs_2 = np.zeros((len(train_samples), 60, 13))
    for i in range(len(train_samples)):
        mfccs_inputs_1[i,:] = mfccs_1[i]
        mfccs_inputs_2[i,:] = mfccs_2[i]

    # Sanity checks.
    print(mfccs_inputs_1.shape)
    print(mfccs_inputs_2.shape)

    # Reshape.
    mfccs_inputs_1 = mfccs_inputs_1.reshape((-1, 60, 13, 1))
    mfccs_inputs_2 = mfccs_inputs_2.reshape((-1, 60, 13, 1))

    # Check if same speaker.
    y = np.equal(speaker_sample1, speaker_sample2)

    # Get sum of how many speakers were same and adjust weigths.
    weight_zero = int(np.sum(y))/int(len(train_samples)-np.sum(y)) # For not same speaker 0s.
    weight_one = 1 # For same speaker 1s.

    # Define batch size.
    batch_size = len(mfccs_inputs_1)//40

    # If Model is passed, dont build new.
    if siamese_model == None:
        # Build Base Model.
        base_model = build_model()

        # Build Siamese NNW.
        siamese_model = build_siamese(base_model,(60,13,1))

    # If verbose is true, print Model summaries.
    if verbose == True:
        base_model.summary()
        siamese_model.summary()

    # Feed MFCC features from samples to Siamese NNW and train it based on
    # if utterances were from same or not same person.

    # Train Model.
    siamese_model.fit(x=[mfccs_inputs_1, mfccs_inputs_2], y=y,
                      validation_split=0.2, batch_size=batch_size,
                      epochs=100, shuffle=True, verbose=2, class_weight={0: weight_zero, 1: weight_one})

    # Return trained Model.
    return siamese_model

# Generate Siamese Models and train them for each set (to be evaluated later). 
# To see Model summaries, set verbose=True (not set true for first Model to see the summary).
siamese_model_basic = train_siamese(train_set_basic, verbose=True)

siamese_model_dependent = train_siamese(train_set_dependent, verbose=False)

siamese_model_limited = train_siamese(train_set_limited, verbose=False)

siamese_model_independent = train_siamese(train_set_independent, verbose=False)

"""If you want to save trained Models to runtime or Drive, run following (after mounting if Drive), change booleans:"""

save_temp = False
save_drive = True

if save_temp == True:
   # Save trained models.
    siamese_model_basic.save("basic_model.h5")
    siamese_model_dependent.save("dependent_model.h5")
    siamese_model_limited.save("limited_model.h5")
    siamese_model_independent.save("independent_model.h5")

if save_drive == True:
  # Save trained models to Drive.
  path = "/content/drive/MyDrive/basic_model.h5"
  siamese_model_basic.save(path)

  path = "/content/drive/MyDrive/dependent_model.h5"
  siamese_model_dependent.save(path)

  path = "/content/drive/MyDrive/limited_model.h5"
  siamese_model_limited.save(path)

  path = "/content/drive/MyDrive/independent_model.h5"
  siamese_model_independent.save(path)

"""To load Models from the runtime or Drive run this block (change booleans):"""

load_temp = False
load_drive = True

if load_temp == True:
  # Load trained models for re-use from runtime.
  siamese_model_basic = tf.keras.models.load_model("basic_model.h5")
  siamese_model_dependent = tf.keras.models.load_model("dependent_model.h5")
  siamese_model_limited = tf.keras.models.load_model("limited_model.h5")
  siamese_model_independent = tf.keras.models.load_model("independent_model.h5")

if load_drive == True:
  # Load trained models for re-use from Drive.
  path = "/content/drive/MyDrive/basic_model.h5"
  siamese_model_basic = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/dependent_model.h5"
  siamese_model_dependent = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/limited_model.h5"
  siamese_model_limited = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/independent_model.h5"
  siamese_model_independent = tf.keras.models.load_model(path)

"""Function for fitting Model with less biased data (samples more evenly spread between same / not same speakers):"""

def train_siamese_equal(train_samples: pd.DataFrame, siamese_model: Model = None, verbose: bool = False):
  
    # Get indices from Sample Set.
    indices = np.arange(len(train_samples))
    np.random.shuffle(indices)
    indices_clone = np.copy(indices)
    np.random.shuffle(indices_clone)

    # Generate Sample Set as pairwise DataFrame based on indices.
    pairwise_map = pd.DataFrame({"sample1": indices, "sample2": indices_clone})

    # Take speakers from Sample Set.
    speaker_sample1 = train_samples["spk"].iloc[pairwise_map["sample1"].values].values
    speaker_sample2 = []

    # Take MFCCs for Sample Set.
    mfccs_1 = train_samples["mfcc"].iloc[speaker_sample1].values

    # Generate Target-scores MFCC scores for half of the utterances from same speaker for each speaker in first MFCC set.
    mfccs_2_target = []
    for i in range(len(speaker_sample1)//2):
        speaker = speaker_sample1[i]
        speaker_sample2.append(speaker)
        mfccs_speaker = train_samples.loc[train_samples["spk"] == speaker, "mfcc"]
        mfccs_target = random.sample(list(mfccs_speaker), 1)
        mfccs_2_target.append(mfccs_target)
    
    # Generate Non-target scores for the other half.
    mfccs_2_non_target = []
    for i in range(len(speaker_sample1)//2):
        speaker = speaker_sample1[i]
        not_speaker = train_samples.loc[train_samples["spk"] != speaker, "spk"].iloc[0]
        speaker_sample2.append(not_speaker)
        mfccs_not_speaker = train_samples.loc[train_samples["spk"] == not_speaker, "mfcc"]
        mfccs_non_target = random.sample(list(mfccs_not_speaker), 1)
        mfccs_2_non_target.append(mfccs_non_target)
    
    # Convert to numpy.
    speaker_sample2 = np.array(speaker_sample2)

    # Combine lists.
    mfccs_2 = mfccs_2_target + mfccs_2_non_target
    mfccs_2 = np.array(mfccs_2)
    
    # Generate correct input format for NNW.
    mfccs_inputs_1 = np.zeros((len(train_samples), 60, 13))
    mfccs_inputs_2 = np.zeros((len(train_samples), 60, 13))
    for i in range(len(train_samples)):
        mfccs_inputs_1[i,:] = mfccs_1[i]
        mfccs_inputs_2[i,:] = mfccs_2[i]

    # Sanity checks.
    print(mfccs_inputs_1.shape)
    print(mfccs_inputs_2.shape)

    # Reshape.
    mfccs_inputs_1 = mfccs_inputs_1.reshape((-1, 60, 13, 1))
    mfccs_inputs_2 = mfccs_inputs_2.reshape((-1, 60, 13, 1))

    # Check if same speaker.
    y = np.equal(speaker_sample1, speaker_sample2)

    # Get sum of how many speakers were same and adjust weigths.
    weight_zero = int(np.sum(y))/int(len(train_samples)-np.sum(y)) # For not same speaker 0s.
    weight_one = 1 # For same speaker 1s.

    # Define batch size.
    batch_size = len(mfccs_inputs_1)//40

    # If Model is passed, dont build new.
    if siamese_model == None:
        # Build Base Model.
        base_model = build_model()

        # Build Siamese NNW.
        siamese_model = build_siamese(base_model,(60,13,1))

    # If verbose is true, print Model summaries.
    if verbose == True:
        base_model.summary()
        siamese_model.summary()

    # Feed MFCC features from samples to Siamese NNW and train it based on
    # if utterances were from same or not same person.

    # Train Model.
    siamese_model.fit(x=[mfccs_inputs_1, mfccs_inputs_2], y=y,
                      validation_split=0.2, batch_size=batch_size,
                      epochs=50, shuffle=True, verbose=2, class_weight={0: weight_zero, 1: weight_one})

    # Return trained Model.
    return siamese_model

# Generate Siamese Models and train them for each set (to be evaluated later). 
# To see Model summaries, set verbose=True (not set true for first Model to see the summary).
siamese_model_basic_equal = train_siamese_equal(train_set_basic, verbose=True)

siamese_model_dependent_equal = train_siamese_equal(train_set_dependent, verbose=False)

siamese_model_limited_equal = train_siamese_equal(train_set_limited, verbose=False)

siamese_model_independent_equal = train_siamese_equal(train_set_independent, verbose=False)

save_temp = False
save_drive = True

if save_temp == True:
   # Save trained models.
    siamese_model_basic_equal.save("basic_model_equal.h5")
    siamese_model_dependent_equal.save("dependent_model_equal.h5")
    siamese_model_limited_equal.save("limited_model_equal.h5")
    siamese_model_independent_equal.save("independent_model_equal.h5")

if save_drive == True:
  # Save trained models to Drive.
  path = "/content/drive/MyDrive/basic_model_equal.h5"
  siamese_model_basic_equal.save(path)

  path = "/content/drive/MyDrive/dependent_model_equal.h5"
  siamese_model_dependent_equal.save(path)

  path = "/content/drive/MyDrive/limited_model_equal.h5"
  siamese_model_limited_equal.save(path)

  path = "/content/drive/MyDrive/independent_model_equal.h5"
  siamese_model_independent_equal.save(path)

load_temp = False
load_drive = True

if load_temp == True:
  # Load trained models for re-use from runtime.
  siamese_model_basic_equal = tf.keras.models.load_model("basic_model_equal.h5")
  siamese_model_dependent_equal = tf.keras.models.load_model("dependent_model_equal.h5")
  siamese_model_limited_equal = tf.keras.models.load_model("limited_model_equal.h5")
  siamese_model_independent_equal = tf.keras.models.load_model("independent_model_equal.h5")

if load_drive == True:
  # Load trained models for re-use from Drive.
  path = "/content/drive/MyDrive/basic_model_equal.h5"
  siamese_model_basic_equal = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/dependent_model_equal.h5"
  siamese_model_dependent_equal = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/limited_model_equal.h5"
  siamese_model_limited_equal = tf.keras.models.load_model(path)

  path = "/content/drive/MyDrive/independent_model_equal.h5"
  siamese_model_independent_equal = tf.keras.models.load_model(path)

# Test basic scenario siamese Model agains target and non-target scores.
def test_basic_scenario_siamese(siamese_model, test_samples, verbose:bool = False):

    # MFCCs for each sample/speaker in Test Set.
    test_mfccs = test_samples["mfcc"]

    # Generate Target scores, 10 x MFCCs from utterances from same speaker for each speaker.
    target_scores = []
    for i in range(len(test_samples)):
        speaker = test_samples["spk"].iloc[i]
        mfccs_speaker = test_samples.loc[test_samples["spk"] == speaker, "mfcc"]
        mfccs_target = random.sample(list(mfccs_speaker), 10)
        for value in mfccs_target:
          target_scores.append(value)

    # Generate Non-target scores 50 x MFCCs from utterances from different speaker for each speaker.
    non_target_scores = []
    for i in range(len(test_samples)):
        speaker = test_samples["spk"].iloc[i]
        mfccs_not_speaker = test_samples.loc[test_samples["spk"] != speaker, "mfcc"]
        mfccs_non_target = random.sample(list(mfccs_not_speaker), 50)
        for value in mfccs_non_target:
          non_target_scores.append(mfccs_non_target)

    # Take MFCCs for Target scores.
    mfccs_targets = np.array(target_scores)
    print(mfccs_targets.shape)

    # Take MFCCs for Non-target scores.
    mfccs_non_targets = np.array(non_target_scores)
    print(mfccs_non_targets.shape)

    # Generate correct input format for NNW for Target scores.
    mfccs_input_targets = np.zeros((mfccs_targets.shape[0], 60, 13))
    for i in range(len(mfccs_targets)):
        mfccs_input_targets[i, :] = mfccs_targets[i]

    # Generate correct input format for NNW for Non-target scores.
    mfccs_input_non_targets = np.zeros((mfccs_nontargets.shape[0], 60, 13))
    for i in range(len(mfccs_nontargets)):
        mfccs_input_non_targets[i, :] = mfccs_nontargets[i]

    # Sanity checks.
    print(mfccs_input_targets.shape)
    print(mfccs_input_non_targets.shape)

    # Reshape.
    mfccs_input_targets = mfccs_input_targets.reshape((-1, 60, 13, 1))
    mfccs_input_non_targets = mfccs_input_non_targets.reshape((-1, 60, 13, 1))

    # Evaluate on loop.
    counter_target = 0
    counter_non_target = 0

    # Y true values.
    y_true_target = np.ones(len(test_mfccs))
    y_true_non_target = np.zeros(len(test_mfccs))

    # Iterate over all MFCC samples in 
    for i in range(len(test_mfccs)):
        
        # Predict with model.
        prediction = siamese_model.predict([test_mfccs[i], mfccs_input_targets[counter_target:counter_target+10]])
        print(prediction)     

        # Evaluation score with Target-scores.
        eval_score = siamese_model.evaluate([test_mfccs[i], mfccs_input_targets[counter_target:counter_target+10]], 
                                            y=[mfccs_input_targets[counter_target:counter_target+10]], verbose=2)
        print(eval_score)

        counter_target += 10

        # Predict with model.
        prediction = siamese_model.predict([test_mfccs[i], mfccs_input_non_targets[counter_non_target:counter_non_target +50]])
        print(prediction)

        # Evaluation score with Non-target-scores.
        eval_score = siamese_model.evaluate([test_mfccs[i], mfccs_input_non_targets[counter_non_target:counter_non_target +50]], 
                                            y=[mfccs_input_non_targets[counter_non_target:counter_non_target +50]], verbose=2)
        print(eval_score)

        counter_non_target += 50

# Test the basic scenario.
test_basic_scenario_siamese(siamese_model_basic, test_set_basic)

"""Here is the Test Module for the 3 other scenarios. The DET-curves are not really usable for the 3 scenarios without modifying the test-scenario fist, as the randomly testing yields almost always only same outcome result (not same speaker) which causes the DET-curves to be always blank as the prediction becomes too easy and there are no false negatives or false positives. Use sampling and verbose to modify sampling and DET-curve creation to better visualize the amount of false-negatives and false positives when the data is not so biased."""

# Test given Siamese Model with given Test Samples.
def test_siamese(siamese_model, test_samples, verbose:bool = False, sampling:bool = False):

    # Define array to take scores and y-scores for DET-curve plotting.
    y_pred_scores = []
    y_true_scores = []

    # Get indices from Test Set.
    indices = np.arange(len(test_samples))
    np.random.shuffle(indices)
    indices_clone = np.copy(indices)
    np.random.shuffle(indices_clone)

    # Generate DataFrame based on indices.
    pairwise_map = pd.DataFrame({"sample1": indices, "sample2": indices_clone})

    # If sampling is false, do random picking.
    if sampling == False:
        # Take speakers from Test Set.
        speaker_sample1 = test_samples["spk"].iloc[pairwise_map["sample1"].values].values
        speaker_sample2 = test_samples["spk"].iloc[pairwise_map["sample2"].values].values

        # Check if same speaker.
        y = np.equal(speaker_sample1, speaker_sample2)
        y_true_scores.append(y)

        # Take MFCCs for Test Set.
        mfccs_1 = test_samples["mfcc"].iloc[speaker_sample1].values
        mfccs_2 = test_samples["mfcc"].iloc[speaker_sample2].values

        # Generate correct input format for NNW.
        mfccs_input_1 = np.zeros((len(test_samples), 60, 13))
        mfccs_input_2 = np.zeros((len(test_samples), 60, 13))
        for i in range(len(test_samples)):
            mfccs_input_1[i, :] = mfccs_1[i]
            mfccs_input_2[i, :] = mfccs_2[i]
    
    # if sampling is True, do more equal sampling for providing DET-curves.
    if sampling == True:

        # Take speakers from Test Set.
        speaker_sample1 = test_samples["spk"].iloc[pairwise_map["sample1"].values].values
        speaker_sample2 = []

        # Take MFCCs for Sample Set.
        mfccs_1 = test_samples["mfcc"].iloc[speaker_sample1].values

        # Generate Target-scores MFCC scores for half of the utterances from same speaker for each speaker in first MFCC set.
        mfccs_2_target = []
        for i in range(len(speaker_sample1)//2):
            speaker = speaker_sample1[i]
            speaker_sample2.append(speaker)
            mfccs_speaker = test_samples.loc[test_samples["spk"] == speaker, "mfcc"]
            mfccs_target = random.sample(list(mfccs_speaker), 1)
            mfccs_2_target.append(mfccs_target)
        
        # Generate Non-target scores for the other half.
        mfccs_2_non_target = []
        for i in range(len(speaker_sample1)//2):
            speaker = speaker_sample1[i]
            not_speaker = test_samples.loc[test_samples["spk"] != speaker, "spk"].iloc[0]
            speaker_sample2.append(not_speaker)
            mfccs_not_speaker = test_samples.loc[test_samples["spk"] == not_speaker, "mfcc"]
            mfccs_non_target = random.sample(list(mfccs_not_speaker), 1)
            mfccs_2_non_target.append(mfccs_non_target)

        # Convert to numpy.
        speaker_sample2 = np.array(speaker_sample2)

        # Check if same speaker.
        y = np.equal(speaker_sample1, speaker_sample2)
        y_true_scores.append(y)

        # Combine lists.
        mfccs_2 = mfccs_2_target + mfccs_2_non_target
        mfccs_2 = np.array(mfccs_2)
        
        # Generate correct input format for NNW.
        mfccs_input_1 = np.zeros((len(test_samples), 60, 13))
        mfccs_input_2 = np.zeros((len(test_samples), 60, 13))
        for i in range(len(test_samples)):
            mfccs_input_1[i,:] = mfccs_1[i]
            mfccs_input_2[i,:] = mfccs_2[i]

        
    # Sanity checks.
    print(mfccs_input_1.shape)
    print(mfccs_input_2.shape)

    # Reshape.
    mfccs_input_1 = mfccs_input_1.reshape((-1, 60, 13, 1))
    mfccs_input_2 = mfccs_input_2.reshape((-1, 60, 13, 1))

    # Predict with model.
    print("Predicting with model on Test Set:")
    prediction = siamese_model.predict([mfccs_input_1, mfccs_input_2])
    print(prediction)
    y_pred_scores.append(prediction)

    # Evaluate model with Test Set.
    print("Evaluating the neural network on Test Set:")
    score = siamese_model.evaluate([mfccs_input_1, mfccs_input_2], y=y, verbose=2)
    print(score)

    # Get and plot DET-curves.
    if verbose == True:

      # Convert to NumPy and transpose to use with DET-curve.
      y_pred_scores = np.array(y_pred_scores)
      y_pred_scores = y_pred_scores.reshape((y_pred_scores.shape[0], y_pred_scores.shape[1]))
      y_true_scores = np.array(y_true_scores)

      y_pred_scores = y_pred_scores.T
      y_true_scores = y_true_scores.T

      # Predicted.
      fprate_pred, fnrate_pred, thresholds_pred = det_curve(y_true=y_true_scores, y_score=y_pred_scores, pos_label=None)
      plt.figure()
      display = DetCurveDisplay(
        fpr=fprate_pred, fnr=fnrate_pred, estimator_name='Prediction estimator')
      display.plot()
      plt.show()

"""Test the 3 specific different Models. Set verbose = True, to get DET-curves if any plot-able data is provided (not in most cases) and sampling = True to use sampling, to provide more evenly spread labels:"""

test_siamese(siamese_model_dependent, test_set_dependent, verbose=False, sampling=False)

test_siamese(siamese_model_limited, test_set_limited, verbose=False, sampling=False)

test_siamese(siamese_model_independent, test_set_independent, verbose=False, sampling=False)

test_siamese(siamese_model_dependent_equal, test_set_dependent, verbose=True, sampling=True)

test_siamese(siamese_model_limited_equal, test_set_limited, verbose=True, sampling=True)

test_siamese(siamese_model_independent_equal, test_set_independent, verbose=True, sampling=True)

"""Test Models with data from other Test Sets."""

# Context dependent with others.
test_siamese(siamese_model_dependent, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_dependent, test_set_limited,verbose=False, sampling=False)
test_siamese(siamese_model_dependent, test_set_independent, verbose=False, sampling=False)

test_siamese(siamese_model_dependent, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_dependent, test_set_limited,verbose=False, sampling=True)
test_siamese(siamese_model_dependent, test_set_independent, verbose=False, sampling=True)

# Context limited with others.
test_siamese(siamese_model_limited, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_limited, test_set_dependent, verbose=False, sampling=False)
test_siamese(siamese_model_limited, test_set_independent, verbose=False, sampling=False)

test_siamese(siamese_model_limited, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_limited, test_set_dependent, verbose=False, sampling=True)
test_siamese(siamese_model_limited, test_set_independent, verbose=False, sampling=True)

# Context independent with others.
test_siamese(siamese_model_independent, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_independent, test_set_dependent, verbose=False, sampling=False)
test_siamese(siamese_model_independent, test_set_limited, verbose=False, sampling=False)

test_siamese(siamese_model_independent, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_independent, test_set_dependent, verbose=False, sampling=True)
test_siamese(siamese_model_independent, test_set_limited, verbose=False, sampling=True)

"""Test Models with Models trained with more equally distributed data."""

# Context dependent with others.
test_siamese(siamese_model_dependent_equal, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_dependent_equal, test_set_limited,verbose=False, sampling=False)
test_siamese(siamese_model_dependent_equal, test_set_independent, verbose=False, sampling=False)

test_siamese(siamese_model_dependent_equal, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_dependent_equal, test_set_limited,verbose=False, sampling=True)
test_siamese(siamese_model_dependent_equal, test_set_independent, verbose=False, sampling=True)

# Context limited with others.
test_siamese(siamese_model_limited_equal, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_limited_equal, test_set_dependent, verbose=False, sampling=False)
test_siamese(siamese_model_limited_equal, test_set_independent, verbose=False, sampling=False)

test_siamese(siamese_model_limited_equal, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_limited_equal, test_set_dependent, verbose=False, sampling=True)
test_siamese(siamese_model_limited_equal, test_set_independent, verbose=False, sampling=True)

# Context independent with others.
test_siamese(siamese_model_independent_equal, test_set_basic, verbose=False, sampling=False)
test_siamese(siamese_model_independent_equal, test_set_dependent, verbose=False, sampling=False)
test_siamese(siamese_model_independent_equal, test_set_limited, verbose=False, sampling=False)

test_siamese(siamese_model_independent_equal, test_set_basic, verbose=False, sampling=True)
test_siamese(siamese_model_independent_equal, test_set_dependent, verbose=False, sampling=True)
test_siamese(siamese_model_independent_equal, test_set_limited, verbose=False, sampling=True)
