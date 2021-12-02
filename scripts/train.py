from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
from models import CNN_Encoder, RNN_Decoder
from argparse import ArgumentParser

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", help="Network architecture in use", choices =["adaptive", "spatial"])
    parser.add_argument("resources_path", help="resources path")
    
    return parser.parse_args()


def load_image(image_path):
    img = tf.io.read_file(image_path) 
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224)) # resize img to size that ResNet expects (224x224)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)

    return img, image_path


def calc_max_length(tensor):
    print(max(len(t) for t in tensor))
    return max(len(t) for t in tensor)


def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy') 
  return img_tensor, cap


def loss_function(real, pred):
  mask   = tf.math.logical_not(tf.math.equal(real, 0))
  loss_  = loss_object(real, pred)
  mask   = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target):
  loss = 0
  ht, ct = decoder.reset_state(batch_size=target.shape[0])
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

  with tf.GradientTape() as tape:
      v_i, v_g = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          predictions, ht, ct, _ = decoder(dec_input, v_i, v_g, ht, ct)
          loss += loss_function(target[:, i], predictions)
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients           = tape.gradient(loss, trainable_variables)
  mod_grad = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(trainable_variables, gradients)]
  
  optimizer.apply_gradients(zip(mod_grad, trainable_variables))

  return loss, total_loss




if __name__ == "__main__":

    args           = parse_args()
    adaptive       = args.model
    resources_path = args.resources_path

    name_of_zip    = 'annotations/captions_train2014.json'
    if not os.path.exists(os.path.abspath('..') + '/' + name_of_zip):
      annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                cache_subdir=os.path.abspath('.'),
                                                origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                extract = True)
      annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    else:
      print("Annotation file found!")
      annotation_file = os.path.abspath('..')+'/annotations/captions_train2014.json'
    name_of_zip = 'train2014'
    if not os.path.exists(os.path.abspath('..') + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip,
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                          extract = True)
      PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
      print("Training images found!")
      PATH = os.path.abspath('..')+'/train2014/'

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_captions        = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    num_examples    = 40000
    train_captions  = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]


    image_model     = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')
    new_input       = image_model.input 
    hidden_layer    = image_model.layers[-1].output 

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer) # create model with input and output layers

    encode_train    = sorted(set(img_name_vector))
    image_dataset   = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset   = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)

    # for img, path in tqdm(image_dataset , total=(num_examples / 32), ncols=80, smoothing=0.0):
    #   batch_features = image_features_extract_model(img)
    #   batch_features = tf.reshape(batch_features,
    #                   (batch_features.shape[0], -1, batch_features.shape[3]))

    #   for bf, p in zip(batch_features, path):

    #     path_of_feature = p.numpy().decode("utf-8") 
    #     np.save(path_of_feature, bf.numpy())

    most_freq = 6000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=most_freq,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = calc_max_length(train_seqs)

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)

    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

    np.save(resources_path + "/ALL_captions", train_captions)
    np.save(resources_path + "/img_name_train", img_name_train)
    np.save(resources_path + "/training_captions", cap_train)
    np.save(resources_path + "/img_name_val", img_name_val)
    np.save(resources_path + "/val_captions", cap_val)

    #Hyperparameters
    BATCH_SIZE    = 64
    BUFFER_SIZE   = 1000
    embedding_dim = 512
    units         = 512
    vocab_size    = len(tokenizer.word_index) + 1
    num_steps     = len(img_name_train) // BATCH_SIZE

    # Shape of the vector extracted from ResNetV2 is (49, 2048)
    features_shape           = 2048
    attention_features_shape = 49

    #Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #Initialize model
    encoder     = CNN_Encoder(embedding_dim)
    decoder     = RNN_Decoder(embedding_dim, units, vocab_size, adaptive)
    optimizer   = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    #Checkpoint manager
    checkpoint_path = resources_path + "/" + adaptive + "/train"
    # checkpoint_path = "./curr_resources/spatial_ckpts/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    
    #Training
    loss_plot = []
    EPOCHS = 12

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        loss_plot.append(total_loss / num_steps)

        if epoch % 4 == 0:
          ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
    plt.savefig(resources_path + '/loss.png')