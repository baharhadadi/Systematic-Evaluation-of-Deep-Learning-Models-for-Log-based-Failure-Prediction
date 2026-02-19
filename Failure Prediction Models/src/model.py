"""
This file contains the method architecture code and the train and test procedure
which is coded with the help of tensorflow and keras
There are two functions to facilitate the init and train methods of the class: train_model and build_model
The main class is FailureRecognitionModel. It has three methods: init, train, test.
"""
import tensorflow as tf
from src.utils import BatchGenerator, attention, TransformerBlock, PositionEmbedding
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from csv import writer
import os
from sklearn.utils import class_weight
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

# builds the model with the help of tensorflow layers
def build_model(dl_encoder, embedding_strategy, max_input_len, embed_len, dropout= 0.2):
    # input of the model
    if embedding_strategy == "BERT":
      inputs = tf.keras.layers.Input(shape=(max_input_len, embed_len))
      x = inputs

    elif embedding_strategy == "Logkey2vec":
      inputs = tf.keras.layers.Input(shape=(max_input_len))
      x = tf.keras.layers.Embedding(200, embed_len, input_length = max_input_len)(inputs)

    else:
      print("the name of embedding strategy is not correct")

    if dl_encoder == "CNN":
      # the first parallel conv layer
      tower_1 = tf.keras.layers.Conv1D(20, 5, padding='same', activation='relu')(x)
      tower_1 = tf.keras.layers.MaxPooling1D(2, strides=1, padding='same')(tower_1)

      # the second parallel conv layer
      tower_2 = tf.keras.layers.Conv1D(20, 10, padding='same', activation='relu')(x)
      tower_2 = tf.keras.layers.MaxPooling1D(2, strides=1, padding='same')(tower_2)
    
      # the third parallel conv layer
      tower_3 = tf.keras.layers.Conv1D(20, 20, padding='same', activation='relu')(x)
      tower_3 = tf.keras.layers.MaxPooling1D(2, strides=1, padding='same')(tower_3)

      # concatinate the output of parallel conv layers
      merged = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
      x = tf.keras.layers.Flatten()(merged)

    elif dl_encoder == "BiLSTM":
      dropout = dropout  # the dropout percentage in the FFN part of the model

      num_heads = 12  # Number of attention heads
      ff_dim = 528  # Hidden layer size in feed forward network inside transformer
      # the shape of the input is the maximum number of the input tokens as max input len and
      # the dimension of each input token which is embed_len since each token passes the embedding phase
      # before fitting into the model

      x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_len, return_sequences=True))(x)

      x = attention(return_sequences=True)(x)

      #x = tf.keras.layers.Dropout(dropout)(x)
      x = tf.keras.layers.Flatten()(x)

    elif dl_encoder == "LSTM":

      x = tf.keras.layers.LSTM(128,  activation="relu")(x)

    elif dl_encoder == "transformer":

      num_heads = 12  # Number of attention heads
      ff_dim = 528 # Hidden layer size in feed forward n528work inside transformer
      # the shape of the input is the maximum number of the input tokens as max input len and
      # the dimension of each input token which is embed_len since each token passes the embedding phase
      # before fitting into the model
      # the transformer blocks builds the language model part of the model
      transformer_block = TransformerBlock(embed_len, num_heads, ff_dim)
      # before fitting the tokens to the model, they will be added with positional embedding values
      positional_embedding_layer = PositionEmbedding(max_input_len, 2000, embed_len)
      # first the inputs passes through positional embedding layer
      x = positional_embedding_layer(x)
      # then they will enter the transformer block(s)
      x = transformer_block(x)
      # the output of the last transformer block will be fitted into a feedforward network which is
      # consist of a global average pooling, dropout, Dense layer with relu activation,
      # dropout and then two dimension dense layer with softmax activation function which will score the input
      # whether its failure or normal sequence
      x = tf.keras.layers.GlobalAveragePooling1D()(x)
 
    # feed forward layer with dropout
    #x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    #x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    # loss function for calculating the loss value used for training and evaluation
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = 'adam'  # choose optimization algorithm for training the model
    # model will be built here
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # compiles the model with its loss function, optimization algorithm and evaluation metrics
    model.compile(loss=loss_object, metrics=['accuracy'],
                  optimizer=optimizer)

    print(model.summary())
    return model


# function for training phase of the model
def train_model(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                model, epoch_num, class_weights, model_name=None):
    # add checkpoint
    filepath = model_name
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max',
                                                    save_weights_only=True)
    # add stop early conditions
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    # set the callbacks list to stop the training sooner whenever the condition is meet
    callbacks_list = [checkpoint, early_stop]

    # start training
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        class_weight = class_weights,
                        shuffle=True
                        )
    # returns the trained model
    return model


# main class for the model design, train, and testing
class FailureRecognitionModel:
    # initialize the hyperparameter values of the model and build it
    def __init__(self, dl_encoder, embedding_strategy, max_input_len, dropout=0.2):
        self.embed_len = 768
        self.max_len = max_input_len
        self.dropout = dropout
        self.dl_encoder = dl_encoder
        self.embedding_strategy = embedding_strategy
        self.model = build_model(self.dl_encoder, self.embedding_strategy, self.max_len, self.embed_len, self.dropout)
        return

    # the method that starts training phase of the model
    def train(self, X, Y, batch_size, epoch_num):
        # store start time 

        split_rate = 0.2

        # split 80 percent of the training set for validation during training
        train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size= split_rate, random_state=42)

        while sum(train_y) == 0 or sum(val_y) == 0:
            train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size= split_rate)

        # oversampling of minority class
        # store the minority class indexes to use for ovesampling
        minority_index = []
        for i in range(len(train_x)):
          if train_y[i]:
            minority_index.append(i)
        
        minority_index = np.array(minority_index)

        # calculate the difference between minority and majority class
        majority_class_size = len(train_x)-len(minority_index)
        k = abs( majority_class_size - len(minority_index))

        # oversampling the minority class to make the class balance
        oversampled_minority = np.random.choice(minority_index, k, replace=True)

        # make it to list to append oversmapled members easily
        train_x = train_x.tolist()
        train_y = train_y.tolist()
        # add ovesampled members
        for i in oversampled_minority:
          train_x.append(train_x[i])
          train_y.append(train_y[i])

        # shuffle the training set again
        (train_x, train_y) = shuffle(train_x, train_y)

        # convert them to np array
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        # generates the batch files for training
        num_train_samples = len(train_x)
        num_val_samples = len(val_x)
        training_generator = BatchGenerator(train_x, train_y, batch_size, self.embed_len, self.max_len, self.embedding_strategy)
        validate_generator = BatchGenerator(val_x, val_y, batch_size, self.embed_len, self.max_len, self.embedding_strategy)

        #print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples, num_val_samples))
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(train_y),
                                                 y = train_y)

        class_weights = dict(enumerate(class_weights))
        # train the model with train and validate generator that generates the batch files and the other parameters
        self.model = train_model(training_generator, validate_generator, num_train_samples, num_val_samples,
                                 batch_size, self.model, epoch_num, class_weights, model_name='model')

    # test the performance of the model with the test dataset
    def test(self, test_x, test_y, batch_size, results_name, results_loc, dataset_folder):
        #print(test_x)
        #tf.config.run_functions_eagerly(True)
        # trim the test dataset to fit the batch size
        test_x = test_x[: len(test_x) // batch_size * batch_size]
        test_y = test_y[: len(test_y) // batch_size * batch_size]
        # batch generator for the test dataset
        test_loader = BatchGenerator(test_x, test_y, batch_size, self.embed_len, self.max_len, self.embedding_strategy)

        # make prediction for the output value of the test dataset with the model
        prediction = self.model.predict_generator(test_loader, steps=(len(test_x) // batch_size), workers=16,
                                                  max_queue_size=32,
                                                  verbose=1)

        # extract predicted the label ID, 0 for normal and 1 for failure, from the output of the last layer
        prediction = np.argmax(prediction, axis=1)

        # check the size of the predicted data
        test_y = test_y[:len(prediction)]
        # the evaluation analysis of the model for test dataset
        report = classification_report(np.array(test_y), prediction, output_dict = True)
        print(report)

        # create new row of results for this dataset
        row = [round(report['1.0']['precision'], 3), round(report['1.0']['recall'],3), round(report['1.0']['f1-score'],3),
         round(report['0.0']['precision'], 3), round(report['0.0']['recall'],3), round(report['0.0']['f1-score'],3)]
        # extract the name of the dataset to be added to the row
        _ , sample_name  = os.path.split(dataset_folder)
        # add the dataset name to the row
        sample_combination = list(map(int, sample_name.split("_")))
        row = sample_combination + row

        # check if the name of csv file has been specified otherwise would create the file
        if results_name =="":
          results_name = sample_name+ '.csv'

        # if the results_loc is not specified, it would be save in current directory
        if results_loc =="":
          results_loc == os.getcwd()

        else:
          # check if the folder of saving the results exist
          if not os.path.exists(results_loc):
            # if not make it
            os.makedirs(results_loc)
          
        # the path of the results csv file
        results_file = os.path.join(results_loc, results_name)

        if not os.path.exists(results_file):
          # create the header of csv file
          pd.DataFrame(columns= ["PF", "MLSL", "DS", "FPerc", "F_P", "F_R", "F_F1", "N_P", "N_R", "N_F1"]).to_csv(results_file, index=None)
          
        # add the result row to the csv file
        with open(results_file, 'a', newline='') as f_object:  
          # Pass the CSV  file object to the writer() function
          writer_object = writer(f_object)
          # Pass the data in the list as an argument into the writerow() function
          writer_object.writerow(row)  
          # Close the file object
          f_object.close()
        return
