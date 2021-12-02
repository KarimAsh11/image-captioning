import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from PIL import Image
from models import CNN_Encoder, RNN_Decoder
from argparse import ArgumentParser




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", help="Network architecture in use", choices =["adaptive", "spatial"])
    parser.add_argument("eval_mode", help="Evaluation mode", choices =["local", "web", "coco"])
    parser.add_argument("resources_path", help="resources path")
    parser.add_argument("--img_path", help="image path or url", required=False)
    
    return parser.parse_args()

def load_pickle(pickle_file):
	"""
	load pickle.

	params:
	    pickle_file (pickle)
	"""
	with open(pickle_file, 'rb') as h:
		return pickle.load(h)


def load_image(image_path):
    img = tf.io.read_file(image_path) 
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, (224, 224)) # resize img to size that ResNet expects (224x224) (from documentation)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    
    return img, image_path

def evaluate(image, adaptive, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))
    beta_plot      = np.zeros((max_length, 1))

    ht, ct = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    v_i, v_g = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, ht, ct, attention_weights = decoder(dec_input, v_i, v_g, ht, ct)
        beta = attention_weights[0, -1]
        
        if adaptive == "adaptive":
            attention_weights = attention_weights[0, 0:-1]

        beta_plot[i]      = beta.numpy()
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot, beta_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    beta_plot      = beta_plot[:len(result), :]
    attention_plot = attention_plot[:len(result), :]


    return result, attention_plot, beta_plot


def plot_attention(image, result, attention_plot, beta_plot, adaptive):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)

        if adaptive == "adaptive":
            title = '%.3f'%(beta_plot[l]) + " - " + result[l]
        else:
            title = result[l]

        ax.set_title(title)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
    plt.savefig("attention_plot.png")


def plot_beta(beta_plot, result):
    len_result = len(result)
    betas = []
    for l in range(len_result): 
        betas.append(float('%.3f'%(beta_plot[l])))

    x=range(len_result)
    fig = plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks(x) 
    ax.set_xticklabels(["{:s}".format((v)) for v in result]) 
    ax.plot(x, betas, linestyle='--', marker='o')  

    fig.canvas.draw()
    plt.ylabel('Beta')
    plt.show() 
    plt.savefig("beta_plot.png")



def eval_local(img_path, adaptive, tokenizer):
    result, attention_plot, beta_plot = evaluate(img_path, adaptive, tokenizer)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(img_path, result, attention_plot, beta_plot, adaptive)
    if adaptive == "adaptive":
        plot_beta(beta_plot, result)

    im = Image.open(img_path)
    im.show()


def eval_web(img_path, adaptive, tokenizer):
    image_url = img_path
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image10'+image_extension, origin=image_url)

    result, attention_plot, beta_plot = evaluate(image_path, adaptive, tokenizer)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot, beta_plot, adaptive)
    if adaptive == "adaptive":
        plot_beta(beta_plot, result)

    im = Image.open(image_path)
    im.show()


def eval_coco(resources_path, adaptive, tokenizer):
    img_name_val = np.load(resources_path+ "/img_name_val.npy")
    cap_val      = np.load(resources_path+ "/val_captions.npy")
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    print("Image selected:", image)
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot, beta_plot = evaluate(image, adaptive, tokenizer)

    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot, beta_plot, adaptive)
    if adaptive == "adaptive":
        plot_beta(beta_plot, result)

    with open((resources_path + "/record_file.txt"), "a") as r:
        r.write(image+"\n")
        r.write(real_caption+"\n")
        r.write(' '.join(result) + "\n")



if __name__ == "__main__":
    args           = parse_args()
    adaptive       = args.model
    mode           = args.eval_mode
    resources_path = args.resources_path
    img_path       = args.img_path

    # #load index to word dict
    tokenizer = load_pickle(resources_path + "/tokenizer.pickle")
    vocab_size = len(tokenizer.word_index) + 1

    #Hyperparameters
    max_length = 49
    embedding_dim = 512
    units = 512

    # Shape of the vector extracted from ResNetV2 is (49, 2048)
    features_shape = 2048
    attention_features_shape = 49

    #ResNet
    name_of_h5_file = 'resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_h5_file):
        image_model  = tf.keras.applications.ResNet152V2(include_top=False)
        new_input    = image_model.input 
        hidden_layer = image_model.layers[-1].output

        image_features_extract_model = tf.keras.Model(new_input, hidden_layer) 
    else:
        print("ResNet pre-trained on ImageNet already downloaded. Loading model from disk...")
        with open('model_architecture.json', 'r') as f:
            image_features_extract_model = tf.keras.models.model_from_json(f.read())
            weights = image_features_extract_model.load_weights(name_of_h5_file)    
        print("ResNet Model loaded!")

    #Initialize model
    optimizer = tf.keras.optimizers.Adam()
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size, adaptive)

    #load ckpt
    checkpoint_path = resources_path + "/" + adaptive + "/train"
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    status = ckpt.restore(ckpt_manager.latest_checkpoint)

    if mode == "local":
        eval_local(img_path, adaptive, tokenizer)
    elif mode == "web":
        eval_web(img_path, adaptive, tokenizer)
    elif mode == "coco":
        eval_coco(resources_path, adaptive, tokenizer)

     
