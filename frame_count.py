import os
import cv2
import shutil
def count_frames(videoPath):
    video=cv2.VideoCapture(videoPath)
    totalFrames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return totalFrames

def categorize_videos(input_folder,output_folder_above,output_folder_below,threshold=20):
    os.makedirs(output_folder_above,exist_ok=True)
    os.makedirs(output_folder_below,exist_ok=True)

    for filename in os.listdir(input_folder):
        videoPath=os.path.join(input_folder,filename)
        frameCount=count_frames(videoPath)

        if frameCount>=threshold:
            shutil.move(videoPath,os.path.join(output_folder_above,filename))
        else:
            shutil.move(videoPath,os.path.join(output_folder_below,filename))

categorize_videos("bullying","bullying_above","bullying_below")