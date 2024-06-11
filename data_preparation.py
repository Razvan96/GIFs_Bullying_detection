import numpy as np
import imageio
import cv2
import os
from glob import glob
from os.path import join, basename
from shutil import copyfile
import csv

"""
## Define hyperparameters
"""

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

### Copy Image ###

def copy_bully_img_to_train_folder():
    SOURCE_DIR = 'C:\\Work\\ML\\GIF_Scrapper\\downloaded_gifs\\Selected_bullying\\'
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\new_train_gif_bullying\\'

    for gif_file in glob(join(SOURCE_DIR, '*.gif'))[:100]:
        copyfile(gif_file, join(TARGET_DIR, basename(gif_file)))
        #print(join(TARGET_DIR, basename(gif_file)))

    files = os.listdir(TARGET_DIR)

    for index, file in enumerate(files):
        os.rename(os.path.join(TARGET_DIR, file), os.path.join(TARGET_DIR, '_Bullying'.join([str(index), '.gif'])))

def copy_bully_img_to_test_folder():
    SOURCE_DIR = 'C:\\Work\\ML\\GIF_Scrapper\\downloaded_gifs\\Selected_bullying\\'
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\new_test_gif_bullying\\'

    for gif_file in glob(join(SOURCE_DIR, '*.gif'))[100:125]:
        copyfile(gif_file, join(TARGET_DIR, basename(gif_file)))
        #print(join(TARGET_DIR, basename(gif_file)))

    files = os.listdir(TARGET_DIR)

    for index, file in enumerate(files):
        os.rename(os.path.join(TARGET_DIR, file), os.path.join(TARGET_DIR, '_Bullying'.join([str(index), '.gif'])))

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

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

def get_frames(path):
    frames = load_video(os.path.join("C:\\Work\\ML\\Video_Classification\\test_25\\", path))
    return frames

def to_gif(images, name):
    converted_images = images.astype(np.uint8)
    new_name = name.replace("avi", "gif")
    imageio.mimsave(os.path.join("C:\\Work\\ML\\Video_Classification\\test_25_gif\\", new_name), converted_images, duration=100)
    #return embed.embed_file("animation.gif")

def transform_train_avi_files_to_gif():
    SOURCE_DIR = 'C:\\Work\\ML\\Video_Classification\\train\\'
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\train_100\\'

    for avi_file in glob(join(SOURCE_DIR, '*CricketShot*.avi'))[:100]:
        copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    for avi_file in glob(join(SOURCE_DIR, '*PlayingCello*.avi'))[:100]:
        copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    for avi_file in glob(join(SOURCE_DIR, '*Punch*.avi'))[:100]:
        copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    for avi_file in glob(join(SOURCE_DIR, '*ShavingBeard*.avi'))[:100]:
        copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    for avi_file in glob(join(SOURCE_DIR, '*TennisSwing*.avi'))[:100]:
        copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    files = os.listdir(TARGET_DIR)

    for file in enumerate(files):
        video = str(file[1])
        frames = get_frames(video)
        to_gif(frames[:MAX_SEQ_LENGTH], video)

def transform_test_avi_files_to_gif():
    SOURCE_DIR = 'C:\\Work\\ML\\Video_Classification\\test\\'
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\test_25\\'

    # for avi_file in glob(join(SOURCE_DIR, '*CricketShot*.avi'))[:25]:
    #     copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))
    #
    # for avi_file in glob(join(SOURCE_DIR, '*PlayingCello*.avi'))[:25]:
    #     copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))
    #
    # for avi_file in glob(join(SOURCE_DIR, '*Punch*.avi'))[:25]:
    #     copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))
    #
    # for avi_file in glob(join(SOURCE_DIR, '*ShavingBeard*.avi'))[:25]:
    #     copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))
    #
    # for avi_file in glob(join(SOURCE_DIR, '*TennisSwing*.avi'))[:25]:
    #     copyfile(avi_file, join(TARGET_DIR, basename(avi_file)))

    files = os.listdir(TARGET_DIR)

    for file in enumerate(files):
        video = str(file[1])
        frames = get_frames(video)
        to_gif(frames[:MAX_SEQ_LENGTH], video)

def create_csv_file_from_train():
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\new_train_gif_bullying\\'
    files = os.listdir(TARGET_DIR)

    with open('new_train_gif.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["video_name", "tag"]

        writer.writerow(field)

        for file in enumerate(files):
            if "CricketShot" in str(file[1]):
                writer.writerow([str(file[1]), "CricketShot"])
            elif "PlayingCello" in str(file[1]):
                writer.writerow([str(file[1]), "PlayingCello"])
            elif "Punch" in str(file[1]):
                writer.writerow([str(file[1]), "Punch"])
            elif "ShavingBeard" in str(file[1]):
                writer.writerow([str(file[1]), "ShavingBeard"])
            elif "TennisSwing" in str(file[1]):
                writer.writerow([str(file[1]), "TennisSwing"])
            elif "Bullying" in str(file[1]):
                writer.writerow([str(file[1]), "Bullying"])

def create_csv_file_from_test():
    TARGET_DIR = 'C:\\Work\\ML\\Video_Classification\\new_test_gif_bullying\\'
    files = os.listdir(TARGET_DIR)

    with open('new_test_gif.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["video_name", "tag"]

        writer.writerow(field)

        for file in enumerate(files):
            if "CricketShot" in str(file[1]):
                writer.writerow([str(file[1]), "CricketShot"])
            elif "PlayingCello" in str(file[1]):
                writer.writerow([str(file[1]), "PlayingCello"])
            elif "Punch" in str(file[1]):
                writer.writerow([str(file[1]), "Punch"])
            elif "ShavingBeard" in str(file[1]):
                writer.writerow([str(file[1]), "ShavingBeard"])
            elif "TennisSwing" in str(file[1]):
                writer.writerow([str(file[1]), "TennisSwing"])
            elif "Bullying" in str(file[1]):
                writer.writerow([str(file[1]), "Bullying"])

copy_bully_img_to_train_folder()
copy_bully_img_to_test_folder()
#transform_train_avi_files_to_gif()
#transform_test_avi_files_to_gif()
create_csv_file_from_train()
create_csv_file_from_test()


