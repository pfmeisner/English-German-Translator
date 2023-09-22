#!/usr/bin/env python
# coding: utf-8

# # Capstone Project
# ## Neural translation model
# ### Instructions
# 
# In this notebook, you will create a neural network that translates from English to German. You will use concepts from throughout this course, including building more flexible model architectures, freezing layers, data processing pipeline and sequence modelling.
# 
# This project is peer-assessed. Within this notebook you will find instructions in each section for how to complete the project. Pay close attention to the instructions as the peer review will be carried out according to a grading rubric that checks key parts of the project instructions. Feel free to add extra cells into the notebook as required.
# 
# ### How to submit
# 
# When you have completed the Capstone project notebook, you will submit a pdf of the notebook for peer review. First ensure that the notebook has been fully executed from beginning to end, and all of the cell outputs are visible. This is important, as the grading rubric depends on the reviewer being able to view the outputs of your notebook. Save the notebook as a pdf (File -> Download as -> PDF via LaTeX). You should then submit this pdf for review.
# 
# ### Let's get started!
# 
# We'll start by running some imports, and loading the dataset. For this project you are free to make further imports throughout the notebook as you wish. 

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import unicodedata
import re


# ![Flags overview image](data/germany_uk_flags.png)
# 
# For the capstone project, you will use a language dataset from http://www.manythings.org/anki/ to build a neural translation model. This dataset consists of over 200,000 pairs of sentences in English and German. In order to make the training quicker, we will restrict to our dataset to 20,000 pairs. Feel free to change this if you wish - the size of the dataset used is not part of the grading rubric.
# 
# Your goal is to develop a neural translation model from English to German, making use of a pre-trained English word embedding module.

# In[2]:


# Run this cell to load the dataset

NUM_EXAMPLES = 20000
data_examples = []
with open('data/deu.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        if len(data_examples) < NUM_EXAMPLES:
            data_examples.append(line)
        else:
            break


# In[3]:


# These functions preprocess English and German sentences

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"ü", 'ue', sentence)
    sentence = re.sub(r"ä", 'ae', sentence)
    sentence = re.sub(r"ö", 'oe', sentence)
    sentence = re.sub(r'ß', 'ss', sentence)
    
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-z?.!,']+", " ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    return sentence.strip()


# #### The custom translation model
# The following is a schematic of the custom translation model architecture you will develop in this project.
# 
# ![Model Schematic](data/neural_translation_model.png)
# 
# Key:
# ![Model key](data/neural_translation_model_key.png)
# 
# The custom model consists of an encoder RNN and a decoder RNN. The encoder takes words of an English sentence as input, and uses a pre-trained word embedding to embed the words into a 128-dimensional space. To indicate the end of the input sentence, a special end token (in the same 128-dimensional space) is passed in as an input. This token is a TensorFlow Variable that is learned in the training phase (unlike the pre-trained word embedding, which is frozen).
# 
# The decoder RNN takes the internal state of the encoder network as its initial state. A start token is passed in as the first input, which is embedded using a learned German word embedding. The decoder RNN then makes a prediction for the next German word, which during inference is then passed in as the following input, and this process is repeated until the special `<end>` token is emitted from the decoder.

# ## 1. Text preprocessing
# * Create separate lists of English and German sentences, and preprocess them using the `preprocess_sentence` function provided for you above.
# * Add a special `"<start>"` and `"<end>"` token to the beginning and end of every German sentence.
# * Use the Tokenizer class from the `tf.keras.preprocessing.text` module to tokenize the German sentences, ensuring that no character filters are applied. _Hint: use the Tokenizer's "filter" keyword argument._
# * Print out at least 5 randomly chosen examples of (preprocessed) English and German sentence pairs. For the German sentence, print out the text (with start and end tokens) as well as the tokenized sequence.
# * Pad the end of the tokenized German sequences with zeros, and batch the complete set of sequences into one numpy array.

# In[4]:


eng_sentences = []
ger_sentences = []


for inx in range(len(data_examples)):
    split_data = re.split(r'\t',data_examples[inx])
    eng_sentences.append(preprocess_sentence(split_data[0]))
    ger_sentences.append("<start> "+preprocess_sentence(split_data[1])+" <end>")
    
    
# print(eng_sentences[:10])
# print(ger_sentences[:10])
# type(ger_sentences[0])


# In[5]:


from tensorflow.keras.preprocessing.text import Tokenizer

ger_tokenizer = Tokenizer(filters='')
ger_tokenizer.fit_on_texts(ger_sentences)
tokenized_ger_sentences = ger_tokenizer.texts_to_sequences(ger_sentences)


# In[6]:


import random



for rand_inx in random.sample(range(len(data_examples)), 5):
    print("Preprocessed English sentence: " + eng_sentences[rand_inx])
    print("Preprocessed German sentence: " + ger_sentences[rand_inx])
    print("Tokenized German sentence: ", tokenized_ger_sentences[rand_inx])
    print('\n')


# In[7]:


from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np

padded_ger_tokens = pad_sequences(tokenized_ger_sentences, padding='post')


# In[8]:


# padded_ger_tokens.shape


# In[9]:


padded_ger_tokens[:2]


# ## 2. Prepare the data with tf.data.Dataset objects

# #### Load the embedding layer
# As part of the dataset preproceessing for this project, you will use a pre-trained English word embedding module from TensorFlow Hub. The URL for the module is https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1. This module has also been made available as a complete saved model in the folder `'./models/tf2-preview_nnlm-en-dim128_1'`. 
# 
# This embedding takes a batch of text tokens in a 1-D tensor of strings as input. It then embeds the separate tokens into a 128-dimensional space. 
# 
# The code to load and test the embedding layer is provided for you below.
# 
# **NB:** this model can also be used as a sentence embedding module. The module will process each token by removing punctuation and splitting on spaces. It then averages the word embeddings over a sentence to give a single embedding vector. However, we will use it only as a word embedding module, and will pass each word in the input sentence as a separate token.

# In[10]:


# Load embedding module from Tensorflow Hub


model_path = './models/tf2-preview_nnlm-en-dim128_1'
embedding_layer = tf.keras.models.load_model(model_path)

#embedding_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1", 
#                                 output_shape=[128], input_shape=[], dtype=tf.string)


# In[11]:


# Test the layer

embedding_layer(tf.constant(["these", "aren't", "the", "droids", "you're", "looking", "for"])).shape


# You should now prepare the training and validation Datasets.
# 
# * Create a random training and validation set split of the data, reserving e.g. 20% of the data for validation (NB: each English dataset example is a single sentence string, and each German dataset example is a sequence of padded integer tokens).
# * Load the training and validation sets into a tf.data.Dataset object, passing in a tuple of English and German data for both training and validation sets.
# * Create a function to map over the datasets that splits each English sentence at spaces. Apply this function to both Dataset objects using the map method. _Hint: look at the tf.strings.split function._
# * Create a function to map over the datasets that embeds each sequence of English words using the loaded embedding layer/model. Apply this function to both Dataset objects using the map method.
# * Create a function to filter out dataset examples where the English sentence is more than 13 (embedded) tokens in length. Apply this function to both Dataset objects using the filter method.
# * Create a function to map over the datasets that pads each English sequence of embeddings with some distinct padding value before the sequence, so that each sequence is length 13. Apply this function to both Dataset objects using the map method. _Hint: look at the tf.pad function. You can extract a Tensor shape using tf.shape; you might also find the tf.math.maximum function useful._
# * Batch both training and validation Datasets with a batch size of 16.
# * Print the `element_spec` property for the training and validation Datasets. 
# * Using the Dataset `.take(1)` method, print the shape of the English data example from the training Dataset.
# * Using the Dataset `.take(1)` method, print the German data example Tensor from the validation Dataset.

# In[ ]:





# In[12]:


def split_and_embed_sentence(eng_sentence,ger_tokens):
    sentence_split = tf.strings.split(eng_sentence, sep=' ')
    embedded_sentence = embedding_layer(sentence_split)
    return embedded_sentence, ger_tokens


def filter_long_embedding(embedding,ger_tokens):
    return tf.shape(embedding)[0] <= 13

def pad_embedding(embedding, ger_tokens):
    padding_length = int(13 - tf.shape(embedding)[0])
    #padding = tf.constant([[0,padding_length],[0,0]])
    padded_embedding = tf.pad(embedding,[[0,padding_length],[0,0]],'CONSTANT')
    return padded_embedding, ger_tokens

    


# In[13]:


train_size = int(0.8 * len(eng_sentences))

train_eng_setences = eng_sentences[:train_size]
train_ger_tokens = padded_ger_tokens[:train_size]

train_dataset = tf.data.Dataset.from_tensor_slices((train_eng_setences, train_ger_tokens))

train_dataset = train_dataset.map(split_and_embed_sentence)
train_dataset = train_dataset.filter(filter_long_embedding)
train_dataset = train_dataset.map(pad_embedding)
train_dataset = train_dataset.batch(16)

valid_eng_sentences = eng_sentences[train_size:]
valid_ger_tokens = padded_ger_tokens[train_size:]

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_eng_sentences, valid_ger_tokens))

valid_dataset = valid_dataset.map(split_and_embed_sentence)
valid_dataset = valid_dataset.filter(filter_long_embedding)
valid_dataset = valid_dataset.map(pad_embedding)
valid_dataset = valid_dataset.batch(16)


# In[14]:


train_example = train_dataset.take(1)
eng_example = next(iter(train_example))[0]

print( "The shape of the batch of English data in the training set is ", np.array(eng_example.shape))

valid_example = valid_dataset.take(1)
ger_token_example = next(iter(valid_example))[1]

print("An example of a batch of German tokens in the validation set is:")
print(np.array(ger_token_example))


# ## 3. Create the custom layer
# You will now create a custom layer to add the learned end token embedding to the encoder model:
# 
# ![Encoder schematic](data/neural_translation_model_encoder.png)

# You should now build the custom layer.
# * Using layer subclassing, create a custom layer that takes a batch of English data examples from one of the Datasets, and adds a learned embedded ‘end’ token to the end of each sequence. 
# * This layer should create a TensorFlow Variable (that will be learned during training) that is 128-dimensional (the size of the embedding space). _Hint: you may find it helpful in the call method to use the tf.tile function to replicate the end token embedding across every element in the batch._
# * Using the Dataset `.take(1)` method, extract a batch of English data examples from the training Dataset and print the shape. Test the custom layer by calling the layer on the English data batch Tensor and print the resulting Tensor shape (the layer should increase the sequence length by one).

# In[15]:


from tensorflow.keras.layers import Layer

class CustomLayer(Layer):
    
    def __init__(self):
        super(CustomLayer,self).__init__()
        self.end_token = self.add_weight(shape = (1,1,128), initializer='random_normal')
    
    def call(self,inputs):
        #batch_size = inputs.shape[0]
        tiled_end_token = tf.tile(self.end_token, tf.constant([16,1,1]))
        return tf.concat([inputs,tiled_end_token], axis=1)
        


# In[16]:


train_example = train_dataset.take(1)
eng_example = next(iter(train_example))[0]

print( "The shape of the batch of English data in the training set is ", np.array(eng_example.shape))

custom_layer = CustomLayer()
custom_layer_test = custom_layer(eng_example)

print( "The shape of the batch of English data after the custom layer is ", np.array(custom_layer_test.shape))


# In[17]:


eng_example.shape[0]


# ## 4. Build the encoder network
# The encoder network follows the schematic diagram above. You should now build the RNN encoder model.
# * Using the functional API, build the encoder network according to the following spec:
#     * The model will take a batch of sequences of embedded English words as input, as given by the Dataset objects.
#     * The next layer in the encoder will be the custom layer you created previously, to add a learned end token embedding to the end of the English sequence.
#     * This is followed by a Masking layer, with the `mask_value` set to the distinct padding value you used when you padded the English sequences with the Dataset preprocessing above.
#     * The final layer is an LSTM layer with 512 units, which also returns the hidden and cell states.
#     * The encoder is a multi-output model. There should be two output Tensors of this model: the hidden state and cell states of the LSTM layer. The output of the LSTM layer is unused.
# * Using the Dataset `.take(1)` method, extract a batch of English data examples from the training Dataset and test the encoder model by calling it on the English data Tensor, and print the shape of the resulting Tensor outputs.
# * Print the model summary for the encoder network.

# In[106]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, LSTM


# In[107]:


input_shape = (13,128)
input_layer = Input(shape=input_shape)
custom_layer = CustomLayer()(input_layer)
masking_layer = Masking(mask_value=0)(custom_layer)
lstm_layer = LSTM(units=512, return_state=True)(masking_layer)

encoder_model = Model(inputs= input_layer, outputs = [lstm_layer[1], lstm_layer[2]])


# In[108]:


train_example = train_dataset.take(1)
eng_example = next(iter(train_example))[0]

encoded_eng_example = encoder_model(eng_example)
print("The shape of the encoded batch of English data is ", np.array(encoded_eng_example).shape)


# In[109]:


encoder_model.summary()


# ## 5. Build the decoder network
# The decoder network follows the schematic diagram below. 
# 
# ![Decoder schematic](data/neural_translation_model_decoder.png)

# You should now build the RNN decoder model.
# * Using Model subclassing, build the decoder network according to the following spec:
#     * The initializer should create the following layers:
#         * An Embedding layer with vocabulary size set to the number of unique German tokens, embedding dimension 128, and set to mask zero values in the input.
#         * An LSTM layer with 512 units, that returns its hidden and cell states, and also returns sequences.
#         * A Dense layer with number of units equal to the number of unique German tokens, and no activation function.
#     * The call method should include the usual `inputs` argument, as well as the additional keyword arguments `hidden_state` and `cell_state`. The default value for these keyword arguments should be `None`.
#     * The call method should pass the inputs through the Embedding layer, and then through the LSTM layer. If the `hidden_state` and `cell_state` arguments are provided, these should be used for the initial state of the LSTM layer. _Hint: use the_ `initial_state` _keyword argument when calling the LSTM layer on its input._
#     * The call method should pass the LSTM output sequence through the Dense layer, and return the resulting Tensor, along with the hidden and cell states of the LSTM layer.
# * Using the Dataset `.take(1)` method, extract a batch of English and German data examples from the training Dataset. Test the decoder model by first calling the encoder model on the English data Tensor to get the hidden and cell states, and then call the decoder model on the German data Tensor and hidden and cell states, and print the shape of the resulting decoder Tensor outputs.
# * Print the model summary for the decoder network.

# In[110]:


from tensorflow.keras.layers import Embedding, Dense

class DecoderModel(Model):
    
    def __init__(self, num_tokens):
        super(DecoderModel,self).__init__()
        self.embedding_layer = Embedding(input_dim = num_tokens,output_dim = 128,mask_zero=True)
        self.lstm_layer = LSTM(units=512, return_state = True, return_sequences=True)
        self.dense_layer = Dense(num_tokens)
    
    def call(self,inputs,hidden_state=None,cell_state=None):
        x = self.embedding_layer(inputs)
        x, hidden_state, cell_state = self.lstm_layer(x, initial_state=[hidden_state,cell_state])
        x = self.dense_layer(x)
        
        return x, hidden_state, cell_state
        

        


# In[112]:


train_example = train_dataset.take(1)
eng_example, ger_tokens = next(iter(train_example))

encoded_hidden_state, encoded_cell_state = encoder_model(eng_example)

# print("The shape of the encoded batch of English data is ", np.array(encoded_eng_example).shape)

num_tokens = len(ger_tokenizer.word_index)

decoder_model = DecoderModel(num_tokens)
decoded_ger_tokens, _,_ = decoder_model(ger_tokens, hidden_state=encoded_hidden_state, cell_state=encoded_cell_state)

print("The decoded German tokens have shape ", decoded_ger_tokens.shape)



# In[113]:


decoder_model.summary()


# ## 6. Make a custom training loop
# You should now write a custom training loop to train your custom neural translation model.
# * Define a function that takes a Tensor batch of German data (as extracted from the training Dataset), and returns a tuple containing German inputs and outputs for the decoder model (refer to schematic diagram above).
# * Define a function that computes the forward and backward pass for your translation model. This function should take an English input, German input and German output as arguments, and should do the following:
#     * Pass the English input into the encoder, to get the hidden and cell states of the encoder LSTM.
#     * These hidden and cell states are then passed into the decoder, along with the German inputs, which returns a sequence of outputs (the hidden and cell state outputs of the decoder LSTM are unused in this function).
#     * The loss should then be computed between the decoder outputs and the German output function argument.
#     * The function returns the loss and gradients with respect to the encoder and decoder’s trainable variables.
#     * Decorate the function with @tf.function
# * Define and run a custom training loop for a number of epochs (for you to choose) that does the following:
#     * Iterates through the training dataset, and creates decoder inputs and outputs from the German sequences.
#     * Updates the parameters of the translation model using the gradients of the function above and an optimizer object.
#     * Every epoch, compute the validation loss on a number of batches from the validation and save the epoch training and validation losses.
# * Plot the learning curves for loss vs epoch for both training and validation sets.
# 
# _Hint: This model is computationally demanding to train. The quality of the model or length of training is not a factor in the grading rubric. However, to obtain a better model we recommend using the GPU accelerator hardware on Colab._

# In[172]:


def decoder_in_out_puts(inputs):
    sliced = inputs[:,1:]
    zeros_tensor = tf.zeros([16, 1], dtype=ger_tokens.dtype)
    outputs = tf.concat([sliced,zeros_tensor], axis=1)
    return inputs, outputs


# In[173]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def forward_backward_pass(eng_in, ger_in, ger_out):
    with tf.GradientTape() as tape:
        eng_hidden, eng_cell = encoder_model(eng_in)
        output_seq, _, _ = decoder_model(ger_in, hidden_state=eng_hidden, cell_state=eng_cell)
        loss_value = loss_object(ger_out, output_seq)
    gradients = tape.gradient(loss_value, encoder_model.trainable_variables + decoder_model.trainable_variables)
    return loss_value, gradients


# In[174]:



def training_loop(train_dataset, val_dataset, num_epochs, optimizer):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_train_loss = tf.keras.metrics.Mean()
        epoch_val_loss = tf.keras.metrics.Mean()
        print("Starting Epoch: ", epoch)
        # Training phase
        for x,y in train_dataset:
            eng_in = x
            ger_in,ger_out = decoder_in_out_puts(y)
            loss_value, gradients = forward_backward_pass(eng_in, ger_in, ger_out)
            optimizer.apply_gradients(zip(gradients, encoder_model.trainable_variables + decoder_model.trainable_variables))
            epoch_train_loss(loss_value)
        print("Finished training phase")
        # Validation phase
        for x,y in val_dataset:
            eng_in = x
            ger_in, get_out = decoder_in_out_puts(y)
            loss_value, _ = forward_backward_pass(eng_in, ger_in, ger_out)
            epoch_val_loss(loss_value)
        print("Finished validation phase")
        # Compute average losses
        train_loss = epoch_train_loss.result()
        val_loss = epoch_valid_loss.result()

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {val_loss:.4f}")

    return train_losses, val_losses


# # Set hyperparameters
# num_epochs = 10
# optimizer = tf.keras.optimizers.Adam()

# # Run the training loop
# train_losses, valid_losses = training_loop(train_dataset, valid_dataset, num_epochs, optimizer)


# In[175]:


# Set hyperparameters
num_epochs = 5
optimizer = tf.keras.optimizers.Adam()

# Run the training loop
train_losses, valid_losses = training_loop(train_dataset, valid_dataset, num_epochs, optimizer)


# ## 7. Use the model to translate
# Now it's time to put your model into practice! You should run your translation for five randomly sampled English sentences from the dataset. For each sentence, the process is as follows:
# * Preprocess and embed the English sentence according to the model requirements.
# * Pass the embedded sentence through the encoder to get the encoder hidden and cell states.
# * Starting with the special  `"<start>"` token, use this token and the final encoder hidden and cell states to get the one-step prediction from the decoder, as well as the decoder’s updated hidden and cell states.
# * Create a loop to get the next step prediction and updated hidden and cell states from the decoder, using the most recent hidden and cell states. Terminate the loop when the `"<end>"` token is emitted, or when the sentence has reached a maximum length.
# * Decode the output token sequence into German text and print the English text and the model's German translation.

# In[ ]:





# In[ ]:





# In[ ]:




