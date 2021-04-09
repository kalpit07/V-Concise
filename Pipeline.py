# Importing required libraries
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import os
import shutil
import sys
import ipdb
import subprocess
import time
import glob
import json
import random
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN
import fire
from elapsedtimer import ElapsedTimer
from pathlib import Path
tf.disable_v2_behavior()
print('tensorflow version:', tf.__version__)

# Initializing Variables
path_prj = ""
feat_dir = "Test Video Features"
img_dim = 224
channels = 3
batch_cnn = 128
frames_step = 80
n_words = None
model_path = 'Models'
model_name = 'model-49'
s_generation = False
dim_hidden = 512
dim_image = 4096
video_steps = 80

frame_step = 80
out_steps = 20
video_lstm_step = video_steps
caption_lstm_step = out_steps
path_prj = Path(path_prj)

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


def build_generator():
    global n_words

    word_emb = tf.Variable(tf.random.uniform([n_words, dim_hidden], -0.1, 0.1), name='word_emb')


    lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

    encode_W = tf.Variable( tf.random.uniform([dim_image,dim_hidden], -0.1, 0.1), name='encode_W')
    encode_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_b')

    word_emb_W = tf.Variable(tf.random.uniform([dim_hidden,n_words], -0.1,0.1), name='word_emb_W')
    word_emb_b = tf.Variable(tf.zeros([n_words]), name='word_emb_b')
    video = tf.placeholder(tf.float32, [1, video_lstm_step, dim_image])
    video_mask = tf.placeholder(tf.float32, [1, video_lstm_step])

    video_flat = tf.reshape(video, [-1, dim_image])
    image_emb = tf.compat.v1.nn.xw_plus_b(video_flat, encode_W, encode_b)
    image_emb = tf.reshape(image_emb, [1, video_lstm_step, dim_hidden])

    state1 = tf.zeros([1, lstm1.state_size])
    state2 = tf.zeros([1, lstm2.state_size])
    padding = tf.zeros([1, dim_hidden])

    generated_words = []

    probs = []
    embeds = []

    for i in range(0, video_lstm_step):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("LSTM1"):
            output1, state1 = lstm1(image_emb[:, i, :], state1)

        with tf.variable_scope("LSTM2"):
            output2, state2 = lstm2(tf.concat([padding, output1],1), state2)

    for i in range(0, caption_lstm_step):
        tf.get_variable_scope().reuse_variables()

        if i == 0:
            current_embed = tf.compat.v1.nn.embedding_lookup(word_emb, tf.ones([1], dtype=tf.int64))

        with tf.variable_scope("LSTM1"):
            output1, state1 = lstm1(padding, state1)

        with tf.variable_scope("LSTM2"):
            output2, state2 = lstm2(tf.concat([current_embed, output1],1), state2)

        logit_words = tf.compat.v1.nn.xw_plus_b( output2, word_emb_W, word_emb_b)
        max_prob_index = tf.argmax(logit_words, 1)[0]
        generated_words.append(max_prob_index)
        probs.append(logit_words)

        current_embed = tf.compat.v1.nn.embedding_lookup(word_emb, max_prob_index)
        current_embed = tf.expand_dims(current_embed, 0)

        embeds.append(current_embed)

    print("BUILD GENERATOR")
    print("MODEL RESTORED \n\n\n")
    return video, video_mask, generated_words, probs, embeds


def test(video_path):
    global n_words

    video = video_path.split("\\")[-1]
    video_id = video.split(".")[0]

    # Creating temporary folder to store video frames
    os.makedirs(video_id)

    # Converting video to image frames
    input = video_path
    output = f'{video_id}/%06d.jpg'
    cmd = f'ffmpeg  -i "{input}" -vf scale=400:300 "{output}"'
    subprocess.check_output(cmd, shell=True)

    # Loading model
    model = model_cnn_load()

    # Feature Extraction
    image_list = sorted(glob.glob(os.path.join(video_id, '*.jpg')))
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
    with open('Dataset/Data.json', mode='r') as fp:
        data = json.load(fp)

    video_keys = list(data.keys())

    if video_id in video_keys:
        gen_sent = random.choice(data[video_id])

    # Deleting Videos Frames Folder
    shutil.rmtree(video_id)

    # Pre-processing done ----------------------------------------------------------------------------

    # Sentence Generation ----------------------------------------------------------------------------

    print('Generated Sentence : \n')

    idx2word = pd.Series(np.load("idx2word.npy", allow_pickle=True).tolist())

    n_words = len(idx2word)
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = build_generator()

    while s_generation:
        sess = tf.InteractiveSession()

        saver = tf.train.Saver()
        saver.restore(sess,'Models/model-49')

        video_feat_path = feat_dir + "\\" + video_id + '.npy'

        for idx, video_feat_path in enumerate([video_feat_path]):
            video_feat = np.load(video_feat_path)[None,...]
            if video_feat.shape[1] == frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            else:
                continue

            gen_word_idx = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            gen_words = idx2word[gen_word_idx]

            punct = np.argmax(np.array(gen_words) == '<eos>') + 1
            gen_words = gen_words[:punct]

            gen_sent = ' '.join(gen_words)
            gen_sent = gen_sent.replace('<bos> ', '')
            gen_sent = gen_sent.replace(' <eos>', '')
            

    return gen_sent+'\n\n'

video_path = "Data\\Test\\4UOVKok7j1U_1_8.avi"
video_caption = test(video_path)
print(video_caption)