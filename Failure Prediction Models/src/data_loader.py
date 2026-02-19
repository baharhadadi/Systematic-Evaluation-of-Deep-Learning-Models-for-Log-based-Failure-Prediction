"""
this file contains the Dataset class for reading the dataset file,
 split it to train and test and get the information of the dataset.
"""
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from csv import reader
import numpy as np
from src.utils import bert_encoder
from src.utils import LogEmbedder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import pickle


class Dataset:
    def __init__(self, dataset_folder, embedding_strategy, log_template_list, use_train_test):
        # dataset contains the tuple of sequence of log template ids and its label
        self.dataset = []
        self.labels = np.array([])
        self.use_train_test = use_train_test
        self.log_template_list = log_template_list
        
        # initialize variables for storing the splited data
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        if not use_train_test:
          # read the raw dataset file
          dataset_loc = os.path.join(dataset_folder, os.path.split(dataset_folder)[1]+".csv")
          with open(dataset_loc, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                if row[0] == "sequence":
                  continue
                # row variable is a list that represents a row in csv
                sequence = row[0]
                label = row[1]
                # convert the string of the log sequence to list of log template Ids
                sequence = list(sequence.replace(" ", "").strip(')').strip('(').split(','))
                s = [int(i[1:-1]) for i in sequence]
                s = np.array(s)
                self.dataset.append(s)
                label = 1 if label == 'failure' else 0
                self.labels = np.append(self.labels, label)
          
          self.dataset = np.array(self.dataset, dtype=object)

          self.split()
        
        else:
          # read from splited dataset
          filename = os.path.join(dataset_folder, "train_x.txt")
          with open(filename, 'rb') as file_object:
            train_x = pickle.load(file_object)

          filename = os.path.join(dataset_folder, "test_x.txt")
          with open(filename, 'rb') as file_object:
            test_x = pickle.load(file_object)

          filename = os.path.join(dataset_folder, "train_y.txt")
          with open(filename, 'rb') as file_object:
            train_y = pickle.load(file_object)

          filename = os.path.join(dataset_folder, "test_y.txt")
          with open(filename, 'rb') as file_object:
            test_y = pickle.load(file_object)
            
          # to have the whole dataset concatinate the train and test
          self.dataset = np.concatenate([train_x , test_x])
          self.labels = np.concatenate([train_y , test_y])

          # load the splited data
          self.x_train = train_x
          self.x_test = test_x
          self.y_train = train_y
          self.y_test = test_y
          
        # if the embedding strategy is BERT, textual data of log template IDs should be processed
        if embedding_strategy == "BERT":
          #process the data
          self.process_data_BERT()

        # if the embedding strategy is fasttext+if-idf, textual data of log template IDs should be processed
        elif embedding_strategy == "Fasttext+tf-idf":
          #process the data
          self.process_data_Fast()

        return
    
    def process_data_Fast(self):
        
        # embedding procedure for each log template
        self.id2text_dict = {}
        with open(self.log_template_list, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                log_template_id = str(row[0])
                text = row[1]
                self.id2text_dict[log_template_id] = text

        # process train_x, train_y, test_x, and test_y seperately
        # now each sequence of log template id will be converted to sequence of log template embedding vectors 
        processed_train_x = []
        for seq in self.x_train:
          full_seq = []
          for log_template_id in seq:
            text = self.id2text_dict[str(log_template_id)]
            full_seq.append(text)
          processed_train_x.append(full_seq)

        embedder = LogEmbedder(processed_train_x, 300)

        embedded_data = []
        for seq in processed_train_x:
          # preprocessed log sequence will be added to the embedded data
          embedded_data.append(embedder.embed(seq))
        self.x_train = np.array(embedded_data,dtype= object)

        processed_test_x = []
        for seq in self.x_test:
          full_seq = []
          for log_template_id in seq:
            text = self.id2text_dict[str(log_template_id)]
            full_seq.append(text)
          processed_test_x.append(full_seq)

        embedded_data = []
        for seq in processed_test_x:
          # preprocessed log sequence will be added to the embedded data
          embedded_data.append(embedder.embed(seq))
        self.x_test = np.array(embedded_data,dtype= object)

        return

    def process_data_BERT(self):
        # in case that the textual informaiton of log sequences are needed
        
        # embedding procedure for each log template

        self.embedded_log_templates_dict = {}   # dictionary that maps each log template ID to its embedding vector

        # id2text_dict is a dictionary that map each log template id to its text
        self.id2text_dict = {}
        with open(self.log_template_list, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                log_template_id = str(row[0])
                text = row[1]
                self.id2text_dict[log_template_id] = text

        # bert encoder embed each log template text
        for t in self.id2text_dict.values():
            if t not in self.embedded_log_templates_dict.keys():
                self.embedded_log_templates_dict[t] = bert_encoder(t)
            
        # process train_x, train_y, test_x, and test_y seperately
        # now each sequence of log template id will be converted to sequence of log template embedding vectors
        processed_train_x  = []
        for seq in self.x_train:
            embedding_list = []
            for log_template_id in seq:
                text = self.id2text_dict[str(log_template_id)]
                embedding_list.append(self.embedded_log_templates_dict[text])
            # preprocessed log sequence will be added to the embedded data
            embedding_list = np.array(embedding_list)
            processed_train_x.append(embedding_list)
        self.x_train = np.array(processed_train_x, dtype=object)

        processed_test_x  = []
        for seq in self.x_test:
            embedding_list = []
            for log_template_id in seq:
                text = self.id2text_dict[str(log_template_id)]
                embedding_list.append(self.embedded_log_templates_dict[text])
            # preprocessed log sequence will be added to the embedded data
            embedding_list = np.array(embedding_list)
            processed_test_x.append(embedding_list)
        self.x_test = np.array(processed_test_x, dtype=object)

        return

    # splits the dataset to test and train
    def split(self):
        split_rate = 0.2
        # first shuffles the data
        (self.dataset, self.labels) = shuffle(self.dataset, self.labels)
        # calculate the training set size
        num_train = int(split_rate * self.dataset.shape[0])
        # split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels, test_size= split_rate, random_state=42)
        # make sure that there is failure sequences in the train set for extreme cases
        while sum(self.y_train) == 0 or sum(self.y_test) == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels, test_size= split_rate)

        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_split_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_size(self):
        return self.dataset.shape[0]

    # outputs the failure and normal number of whole dataset as well as training and testing data
    def get_label_distribution(self):

        failure_num = sum([1 for i in self.labels if i == 1])
        normal_num = len(self.dataset) - failure_num
        print("the number of failure sequences is {} and the number of normal sequences is {},"
              "the total percentage of failure sequences is {}".format(failure_num, normal_num,
                                                                       failure_num / self.get_size()))

        # checks if the dataset is split to train and test
        if self.x_train:
            train_failure_num = sum([1 for i in self.labels if i == 1])
            train_normal_num = len(self.x_train) - train_failure_num
            print("the number of failure sequences in training set is {} and the number "
                  "of normal sequences in training set is {},"
                  "the total percentage of failure sequences in training set is {}".format(train_failure_num,
                                                                                           train_normal_num,
                                                                                           train_failure_num / len(
                                                                                               self.x_train)))
        if self.x_test:
            test_failure_num = sum([1 for i in self.labels if i == 1])
            test_normal_num = len(self.x_test) - test_failure_num
            print("the number of failure sequences in testing set is {} and the number"
                  " of normal sequences in testing set is {},"
                  "the total percentage of failure sequences in testing set is {}".format(test_failure_num,
                                                                                          test_normal_num,
                                                                                          test_failure_num / len(
                                                                                              self.x_test)))
        return
