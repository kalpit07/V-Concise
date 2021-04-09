# Importing required libraries
import shutil
import glob
from tqdm import tqdm
import numpy as np
import cv2
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model 
import subprocess

# Initializing variables
video_dest = "Dataset\YouTubeClips"
feat_dir = "Video Features"
temp_dest = "Temp"
img_dim = 224
channels = 3
batch_cnn = 128
frames_step = 80

# Convert the video into image frames at a specified sampling rate 
def video_to_frames(video):
    if os.path.exists(temp_dest):
        print(" cleanup: " + temp_dest + "/")
        shutil.rmtree(temp_dest)
    os.makedirs(temp_dest)
    
    input = video
    output = f'{temp_dest}/%06d.jpg'
    cmd = f'ffmpeg  -i "{input}" -vf scale=400:300 "{output}"'
    subprocess.check_output(cmd, shell=True)

# Load the pre-trained VGG16 Model and extract the dense features as output
def model_cnn_load():
    model = VGG16(weights = "imagenet", include_top=True, input_shape = (img_dim,img_dim,channels))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input,outputs=out)
    return model_final

# Load the video images
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(img_dim,img_dim))
    return img

# Extract the features from the pre-trained CNN 
def extract_feats_pretrained_cnn():
        
    model = model_cnn_load()
    print('Model loaded..!!')
        
    if not os.path.isdir(feat_dir):
        os.mkdir(feat_dir)

    video_list = glob.glob(os.path.join(video_dest, '*.avi'))
    #print(video_list) 
    
    for video in tqdm(video_list):
        
        video_id = video.split("\\")[-1].split(".")[0]
        print(f'Processing video {video}')
        
        video_to_frames(video)

        image_list = sorted(glob.glob(os.path.join(temp_dest, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, frames_step))
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), img_dim, img_dim, channels))
        for i in range(len(image_list)):
            img = load_image(image_list[i])
            images[i] = img
        images = np.array(images)
        fc_feats = model.predict(images, batch_size=batch_cnn)
        img_feats = np.array(fc_feats)
        outfile = os.path.join(feat_dir, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(temp_dest)

# Extracting features
extract_feats_pretrained_cnn()