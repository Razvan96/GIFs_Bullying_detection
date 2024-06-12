from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import random
import csv
import shutil
print(tf.__version__)


source_path="dataset_model_3"
train_ratio=0.8
if os.path.exists("train"):
    shutil.rmtree("train")
if os.path.exists("test"):
    shutil.rmtree("test")
os.mkdir("train")
os.mkdir("test")

video_name_train=[]
video_name_test=[]
tag_train=[]
tag_test=[]

for folderName in os.listdir(source_path):
    subfolderPath=os.path.join(source_path,folderName)
    imageNames=os.listdir(subfolderPath)
    random.shuffle(imageNames)

    for idx,imgName in enumerate(imageNames):
        caleSursaImg=os.path.join(subfolderPath,imgName)
        if idx<train_ratio*len(imageNames):
            video_name_train.append(imgName)
            tag_train.append(imgName[:imgName.find('_')])
            shutil.copy2(caleSursaImg,"train")
        else:
            video_name_test.append(imgName)
            tag_test.append(imgName[:imgName.find('_')])
            shutil.copy2(caleSursaImg,"test")

with open("train.csv","w",newline="") as csv_file:
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow(['video_name','tag'])
    for a,b in zip(video_name_train,tag_train):
        csv_writer.writerow([a,b])

with open("test.csv","w",newline="") as csv_file:
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow(['video_name','tag'])
    for a,b in zip(video_name_test,tag_test):
        csv_writer.writerow([a,b])

IMG_SIZE = 169
EPOCHS = 50
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f'Total videos for testing: {len(test_df)}')


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor')


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype='bool')
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    width_min = 5000
    height_min = 5000

    for idx, path in enumerate(video_paths):
        vcap = cv2.VideoCapture(os.path.join(root_dir, path))

        if vcap.isOpened():
            width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if width_min > width:
                width_min = width
            if height_min > height:
                height_min = height

    print("width min: " + str(width_min))
    print("height min: " + str(height_min))

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype='bool')
        temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return rnn_model


def run_experiment():
    filepath = "./videoclasifier_body_building/checkpoint"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1)
    seq_model = get_sequence_model()
    history = seq_model.fit([train_data[0], train_data[1]],
                            train_labels,
                            validation_split=0.3,
                            epochs=20,
                            callbacks=[checkpoint])
    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()

import numpy as np
import pandas as pd
import cv2
import os
from tensorflow import keras

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 169

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor')


feature_extractor = build_feature_extractor()
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))


def get_sequence_model(nrClase):
    # class_vocab=label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(nrClase, activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return rnn_model


def run_experiment_test(filepath, nrClase):
    seq_model = get_sequence_model(nrClase)
    seq_model.load_weights(filepath)
    return seq_model


sequence_model_1 = run_experiment_test("./videoclasifier_1/checkpoint", 3)
sequence_model_2 = run_experiment_test("./videoclasifier_2/checkpoint", 2)
sequence_model_3 = run_experiment_test("./videoclasifier_3/checkpoint", 2)


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype='bool')
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask


def sequence_prediction(path):
    # class_vocab=label_processor.get_vocabulary()
    model1_classes = ['bodyBuilding', 'bullying', 'waterSports']
    model2_classes = ['kayaking', 'rowing']
    model3_classes = ['pullUps', 'pushUps']

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities_1 = sequence_model_1.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities_1)[::-1]:
        print(f"{model1_classes[i]}: {probabilities_1[i] * 100:5.2f}%")

    maxIndex = np.argmax(probabilities_1)
    if maxIndex == 0:
        probabilities_3 = sequence_model_3.predict([frame_features, frame_mask])[0]
        for i in np.argsort(probabilities_3)[::-1]:
            print(f"{model3_classes[i]}: {probabilities_3[i] * 100:5.2f}%")
    elif maxIndex == 2:
        probabilities_2 = sequence_model_2.predict([frame_features, frame_mask])[0]
        for i in np.argsort(probabilities_2)[::-1]:
            print(f"{model2_classes[i]}: {probabilities_2[i] * 100:5.2f}%")


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
sequence_prediction(test_video)
