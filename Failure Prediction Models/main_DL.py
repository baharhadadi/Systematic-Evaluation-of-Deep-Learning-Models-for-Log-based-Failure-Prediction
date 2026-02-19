"""
This file contains the main part of the automata-based data generator
where the input data is loaded and processed
A model is built based on input arguments and is trained and tested with the input data 
"""

import argparse
from src.data_loader import Dataset
from src.model import FailureRecognitionModel
import pickle 
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dl_encoder', help="the type of the dl encoder use for classification",
                        type=str)
    parser.add_argument('embedding_strategy', help="the type of the embedding strategy use for classification",
                        type=str)
    parser.add_argument('dataset_folder', help="the location of the dataset folder of labeled log sequences in csv file",
                        type=str)
    parser.add_argument('log_template_list', help="the location of the csv file of log template ID to log template text",
                        type=str)
    parser.add_argument('--input_len', help="the maximum length of input sequence of the language"
                                                " model, default is 75",
                        type=int, default=75)
    parser.add_argument('--batch_size', help="the size of batch for training, default is 64",
                        type=int, default=64)
    parser.add_argument('--epoch', help="the number of epochs for training, default is 20",
                        type=int, default=20)
    parser.add_argument('--result_name', help="the name of the csv file to store the results of the model",
                        type=str, default="")
    parser.add_argument('--result_loc', help="the location of the csv file to store the results of the model",
                        type=str, default="")
    parser.add_argument('--use_train_test', help="whether using the raw data or splited data, default is raw data",
                        dest='use_train_test', action='store_true', default=False)          
    args = parser.parse_args()

    # load the dataset from the dataset of log sequences and the dictionary of ids to the log templates' text
    dataset = Dataset(args.dataset_folder, args.embedding_strategy, args.log_template_list, args.use_train_test)

    parent_file_name = args.dataset_folder

    train_x, train_y, test_x, test_y = dataset.get_split_data()

    model = FailureRecognitionModel(args.dl_encoder, args.embedding_strategy, args.input_len)

    model.train(train_x, train_y, args.batch_size, args.epoch)

    model.test(test_x, test_y, args.batch_size, args.result_name, args.result_loc, args.dataset_folder)
