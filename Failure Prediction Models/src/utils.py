"""
This file contains the functions and classes that are used in model.py and data_loader.py files.
The functions are bert_encoder and positional_encoding, and the classes are for layers of the neural netwrok model which are
TransformerBlock class, PositionalEmbedding class as well as BatchGenerator class.
"""
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras import layers
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import backend as K


# tokenization of the text by the bert tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# import the bert model for semantic embedding process of the each log template text
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# function for tokenization and semantic embedding of the log templates
def bert_encoder(s, no_wordpiece=0):
    # remove numbers and whatever doesn't exist in bert tokenizer vocab dictionary 
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    # tokenizes the text
    inputs = bert_tokenizer(s, return_tensors='tf', max_length=512)
    # fit the tokenized text to the bert model to get embedding vector of the log template text
    outputs = bert_model(**inputs)
    # Computes the mean of elements across dimensions of a tensor
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    # return the first output token, the sequence embedding as the log template embedding vector
    return v[0]

class LogEmbedder:
    def __init__(self, log_data, vector_size):
        # Convert log sequences to individual strings
        self.log_strs = [" ".join(seq) for seq in log_data]

        # Train FastText model
        self.model = FastText(sentences=log_data, vector_size=vector_size, window=5, min_count=1, workers=4)

        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        self.tfidf_vectorizer.fit(self.log_strs)

    def embed(self, log_sequence):
        # Convert the log sequence to a single string
        log_str = " ".join(log_sequence)

        # Get TF-IDF weights
        tfidf_weights = self.tfidf_vectorizer.transform([log_str]).toarray()

        # Compute weighted embeddings
        embeddings = []
        for word in log_sequence:
            if word in self.tfidf_vectorizer.vocabulary_:
                word_idx = self.tfidf_vectorizer.vocabulary_[word]
                weight = tfidf_weights[0, word_idx]
                embeddings.append(self.model.wv[word] * weight)
            else:
                embeddings.append(self.model.wv[word])

        return np.array(embeddings)

# this function computes the positional embedding of input sequence that is going to be used by attention layer
def positional_encoding(position, d_model):
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# this class contians the architecture of a transformerBlock which consists of a multihead attention layer,
# normalization and dropout, sequencial layer, and then second normalization and dropout
class TransformerBlock(tf.keras.layers.Layer):
    # initialize the layers
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    # compute the output value of an input passing through the layers of the transformer block
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        #attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        #ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# thid class is for possitional embedding layer of the language model and computes the embedding with respect to 
# the maximum number of input tokens, size of the vocabs from embedding phase and the embedding vector dimention of each input token
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim)
    # adds the input value with its positional embedding
    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        return x

    
# class for generation of batch files during traning phase of the model
# each batch is a set of preprocessed log sequences
class BatchGenerator(tf.keras.utils.Sequence):
   
    # initialize the parameters
    def __init__(self, X, Y, batch_size, embed_dim, max_len, embedding_strategy):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.embedding_strategy = embedding_strategy

    # returns the number of batch files 
    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    # returns batch idx'th file item
    def __getitem__(self, idx):
        # the input size in case of using BERT is a 2D data since embedding matrix is already calculated using pretrained BERT
        if self.embedding_strategy == "BERT" or self.embedding_strategy == "Fasttext+tf-idf":
          # create the batch file with respect to the input index
          x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
          X = np.zeros((len(x), self.max_len, self.embed_dim))
          Y = np.zeros((len(x), 2))
          item_count = 0
          # check the length of each log sequence
          for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            # if the length of the log sequence is greater than the maximum length of the input sequence of the model,
            # it would trim the beginning of the sequence
            if len(x) > self.max_len:
                x = x[-self.max_len:]
            # pad the sequence to fit the input size for the model
            x = np.pad(np.array(x), pad_width=((self.max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            # reshape the sequence to the size of the input of the model
            X[item_count] = np.reshape(x, [self.max_len, self.embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
            
        # in case of using Logkey2vec, the input is 1D since the embedding step happens inside the trainable part of the model
        elif self.embedding_strategy == "Logkey2vec":
          # create the batch file with respect to the input index
          x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
          X = np.zeros((len(x), self.max_len))
          Y = np.zeros((len(x), 2))
          item_count = 0
          # check the length of each log sequence
          for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            # if the length of the log sequence is greater than the maximum length of the input sequence of the model,
            # it would trim the beginning of the sequence
            if len(x) > self.max_len:
                x = x[-self.max_len:]
            # pad the sequence to fit the input size for the model
            x = np.pad(np.array(x), pad_width=((self.max_len - len(x)), (0)), mode='constant',
                       constant_values=0)
            # reshape the sequence to the size of the input of the model
            X[item_count] = np.reshape(x, [self.max_len])
            Y[item_count] = self.Y[i]
            item_count += 1
        # returns the batch file
        return X[:], Y[:, 0]

# the attention layer for BiLSTM network
class attention(tf.keras.layers.Layer):
  def __init__(self, return_sequences=True):
    self.return_sequences = return_sequences
    super(attention,self).__init__()
  
  def build(self, input_shape):
    self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
    self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
    super(attention,self).build(input_shape)
  
  def call(self, x):
    e = K.tanh(K.dot(x,self.W)+self.b)
    a = K.softmax(e, axis=1)
    output = x*a
    if self.return_sequences:
      return output
    return K.sum(output, axis=1)
