import numpy as np
import pandas as pd
import cv2
import os
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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


sequence_model_1 = run_experiment_test("./video_classifier/checkpoint", 3)
#sequence_model_2 = run_experiment_test("./videoclasifier_2/checkpoint", 2)
#sequence_model_3 = run_experiment_test("./videoclasifier_3/checkpoint", 2)


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

    maxIndex = np.argmax(probabilities_1)
    return maxIndex


def all_predictions_main_model():
    true_labels = []
    predicted_labels = []
    for name in test_df["video_name"].values.tolist():
        if 'bodyBuilding' in name:
            true_labels.append(0)
        elif 'bullying' in name:
            true_labels.append(1)
        else:
            true_labels.append(2)
        predicted_labels.append(sequence_prediction(name))

    class_names = ['bodyBuilding', 'bullying', 'waterSports']
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


all_predictions_main_model()