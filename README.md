# GIFs_Bullying_detection
Bullying Detection Solution for GIFs Using a Deep Learning Approach

Our program is based on the code from the following GitHub file: https://github.com/keras-team/keras-io/blob/master/examples/vision/video_classification.py. The main contributions that we came with are:
  *We have implemented at the beginning of our program a functionality that randomly takes the files from the dataset and puts them in the directories for training and testing. The ratio for the former is 80%, while for the latter is 20%.
  *We have employed three Recurrent Neural Networks (RNNs) instead of one as the author utilized. The first RNN classifies the media files into bullying, Water sports, and Bodybuilding. The second can further classify the files into Rowing or Kayaking. The third one can classify the files into Pull Ups and Handstand Pushups.
  *The square that is cropped from each frame of a video has the size of the side changed. We reduced it from 224 to 169.
  *We have increased the number of epochs from 10 to 50.

Many thanks again to [Sayak Paul](https://github.com/sayakpaul) as he aggred to use parts of his code from the above GitHub URL.
