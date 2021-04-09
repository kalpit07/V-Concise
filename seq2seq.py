# Importing required libraries
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import os
import sys
import ipdb
import time
import cv2
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN
import fire
from elapsedtimer import ElapsedTimer
from pathlib import Path
tf.disable_v2_behavior()
print('tensorflow version:', tf.__version__)

# Initializing variables
path_prj = ""
caption_file = "Dataset/Video Corpus/video_corpus.csv"
feat_dir = "Video Features"
cnn_feat_dim = 4096
h_dim = 512
batch_size = 32
lstm_steps = 80
video_steps = 80
out_steps = 20
learning_rate = 1e-4
epochs = 50
frame_step = 80
model_path = 'Models'
model = 'model-49'
n_words = None

dim_image = cnn_feat_dim
dim_hidden = h_dim
video_lstm_step = video_steps
caption_lstm_step = out_steps
path_prj = Path(path_prj)

train_text_path =  caption_file
train_feat_path =  feat_dir

test_text_path =  caption_file
test_feat_path =  feat_dir


def build_model():
    global n_words

    # Defining the weights associated with the Network
    word_emb = tf.Variable(tf.random.uniform([n_words, dim_hidden], -0.1, 0.1), name='word_emb')

    lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    encode_W = tf.Variable( tf.random.uniform([dim_image,dim_hidden], -0.1, 0.1), name='encode_W')
    encode_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_b')
    
    word_emb_W = tf.Variable(tf.random.uniform([dim_hidden,n_words], -0.1,0.1), name='word_emb_W')
    word_emb_b = tf.Variable(tf.zeros([n_words]), name='word_emb_b')
    
    # Placeholders 
    video = tf.placeholder(tf.float32, [batch_size, video_lstm_step, dim_image])
    video_mask = tf.placeholder(tf.float32, [batch_size, video_lstm_step])

    caption = tf.placeholder(tf.int32, [batch_size, caption_lstm_step+1])
    caption_mask = tf.placeholder(tf.float32, [batch_size, caption_lstm_step+1])

    video_flat = tf.reshape(video, [-1, dim_image])
    image_emb = tf.compat.v1.nn.xw_plus_b( video_flat, encode_W,encode_b )         
    image_emb = tf.reshape(image_emb, [batch_size, lstm_steps, dim_hidden])

    state1 = tf.zeros([batch_size, lstm1.state_size])
    state2 = tf.zeros([batch_size, lstm2.state_size])
    padding = tf.zeros([batch_size, dim_hidden])

    probs = []
    loss = 0.0

    #  Encoding Stage 
    for i in range(0, video_lstm_step):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("LSTM1"):
            output1, state1 = lstm1(image_emb[:,i,:], state1)

        with tf.variable_scope("LSTM2"):
            output2, state2 = lstm2(tf.concat([padding, output1],1), state2)

    #  Decoding Stage  to generate Captions 
    for i in range(0, caption_lstm_step):

        with tf.device("/cpu:0"):
            current_embed = tf.compat.v1.nn.embedding_lookup(word_emb, caption[:, i])

        tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("LSTM1"):
            output1, state1 = lstm1(padding, state1)

        with tf.variable_scope("LSTM2"):
            output2, state2 = lstm2(tf.concat([current_embed, output1],1), state2)

        labels = tf.expand_dims(caption[:, i+1], 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat([indices, labels],1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_words]), 1.0, 0.0)

        logit_words = tf.compat.v1.nn.xw_plus_b(output2, word_emb_W, word_emb_b)
        
        # Computing the loss     
        cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logit_words,labels=onehot_labels)
        cross_entropy = cross_entropy * caption_mask[:,i]
        probs.append(logit_words)

        current_loss = tf.reduce_sum(cross_entropy)/batch_size
        loss = loss + current_loss
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)    
    print('BUILD MODEL')
    return loss, video, video_mask, caption, caption_mask, probs,train_op


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
    return video, video_mask, generated_words, probs, embeds


def get_data(text_path,feat_path):
    text_data = pd.read_csv(text_path, sep=',')
    text_data = text_data[text_data['Language'] == 'English']
    text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
    text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
    text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists(x))]
    text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]
    
    unique_filenames = sorted(text_data['video_path'].unique())
    data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
    print('GET DATA')
    return data


def train_test_split(data,test_frac=0.2):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_indices_rec = int((1 - test_frac)*len(data))
    indices_train = indices[:train_indices_rec]
    indices_test = indices[train_indices_rec:]
    data_train, data_test = data.iloc[indices_train],data.iloc[indices_test]
    print(data_train.head())
    print(data_test.head())
    data_train.reset_index(inplace=True)
    data_test.reset_index(inplace=True)
    print('TRAIN TEST SPLIT')
    return data_train,data_test


def get_test_data(text_path,feat_path):
    text_data = pd.read_csv(text_path, sep=',')
    text_data = text_data[text_data['Language'] == 'English']
    text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
    text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
    text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists( x ))]
    text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = sorted(text_data['video_path'].unique())
    test_data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
    print('GET TEST DATA')
    return test_data       
    

def create_word_dict(sentence_iterator, word_count_threshold=5):
    
    word_counts = {}
    sent_cnt = 0
    
    for sent in sentence_iterator:
        sent_cnt += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    
    idx2word = {}
    idx2word[0] = '<pad>'
    idx2word[1] = '<bos>'
    idx2word[2] = '<eos>'
    idx2word[3] = '<unk>'

    word2idx = {}
    word2idx['<pad>'] = 0
    word2idx['<bos>'] = 1
    word2idx['<eos>'] = 2
    word2idx['<unk>'] = 3

    for idx, w in enumerate(vocab):
        word2idx[w] = idx+4
        idx2word[idx+4] = w

    word_counts['<pad>'] = sent_cnt
    word_counts['<bos>'] = sent_cnt
    word_counts['<eos>'] = sent_cnt
    word_counts['<unk>'] = sent_cnt

    print('CREAT WORD DICTION')
    return word2idx,idx2word
    

def train():
    global n_words

    data = get_data(train_text_path, train_feat_path)
    train_data, test_data = train_test_split(data, test_frac=0.2)
    train_data.to_csv(f'{path_prj}/train.csv', index=False)
    test_data.to_csv(f'{path_prj}/test.csv', index=False)

    print(f'Processed train file written to {path_prj}/train_corpus.csv')
    print(f'Processed test file written to {path_prj}/test_corpus.csv')
            

    train_captions = train_data['Description'].values
    test_captions = test_data['Description'].values

    captions_list = list(train_captions)
    captions = np.asarray(captions_list, dtype=np.object)
    print(type(captions))
    print(captions)

    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))

    word2idx, idx2word = create_word_dict(captions, word_count_threshold=0)
    
    np.save(path_prj/ "word2idx", word2idx)
    np.save(path_prj/ "idx2word" , idx2word)
    n_words = len(word2idx)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs, train_op = build_model()
    sess = tf.InteractiveSession()
    
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    tf.global_variables_initializer().run()


    loss_out = open('Model/loss.txt', 'w')
    val_loss = []

    for epoch in range(0, epochs):
        print(f"\nEPOCH Number = {epoch}  ---------------------------------------------------------------------------------------------------")
        val_loss_epoch = []

        index = np.arange(len(train_data))

        train_data.reset_index()
        np.random.shuffle(index)
        train_data = train_data.loc[index]

        current_train_data = train_data.groupby(['video_path']).first().reset_index()


        for start, end in zip(
                range(0, len(current_train_data),batch_size),
                range(batch_size,len(current_train_data),batch_size)):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, video_lstm_step,dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid),current_videos))
            current_feats_vals = np.array(current_feats_vals) 

            current_video_masks = np.zeros((batch_size,video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
            current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))


            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in word2idx:
                        current_word_ind.append(word2idx[word])
                    else:
                        current_word_ind.append(word2idx['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_caption: current_caption_matrix
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            val_loss_epoch.append(loss_val)

            print('Batch starting index: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
            loss_out.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # Draw loss curve every epoch
        
        val_loss.append(np.mean(val_loss_epoch))
        plt_save_dir = "loss_imgs"
        plt_save_img_name = str(epoch) + '.png'
        plt.plot(range(len(val_loss)),val_loss, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name)) 

        if np.mod(epoch,7) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
    print('TRAIN')
    loss_out.close()
    
    
def inference():
    global n_words

    test_data = get_test_data(test_text_path,test_feat_path)
    test_videos = test_data['video_path'].unique()

    idx2word = pd.Series(np.load("idx2word.npy", allow_pickle=True).tolist())

    n_words = len(idx2word)
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess,model_path+'/'+model)

    f = open(f'{path_prj}/video_captioning_results.txt', 'w')
    for idx, video_feat_path in enumerate(test_videos[:15]):
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
        print(f'Video path {video_feat_path} - Generated Caption :')
        print(gen_sent,'\n')
        f.write(video_feat_path + '\n')
        f.write(gen_sent + '\n\n')


# train()
inference()