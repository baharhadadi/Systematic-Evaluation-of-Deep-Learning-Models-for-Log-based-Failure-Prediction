import argparse
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', help="the DL encoder type either CNN, BiLSTM, LSTM, or transformer",
                        type=str)
    parser.add_argument('embedding_strategy', help="the embedding strategy either BERT or Logkey2vec",
                        type=str)
    parser.add_argument('dataset_collection', help="the name of dataset collections either M1, M2, or M3",
                        type=str)                    
    parser.add_argument('--results_loc', help="the location of the folder storing results file",
                        type=str, default = "\Rep_Results")
    parser.add_argument('--use_train_test', help="whether to use splited data or raw data, default is raw",
                        dest='use_train_test', action='store_true', default=True)
    args = parser.parse_args()

    # read the level combination
    combination = "DataSampleCombinations.csv"
    df = pd.read_csv(combination, header = None)
    
    # set to location of dataset which is either from raw folder or train_test folder
    datasetfolder = "Datasets/raw/" + args.dataset_collection
    
    if args.use_train_test:
      datafolder = "Datasets/train_test/" + args.dataset_collection

    # for on the row of level combinations
    for index, row in df.iterrows():

      #find its generated dataset
      dataset_folder = "{}_{}_{}_{}".format(row[1][-1],row[2],row[3],row[4])
      parent_path = os.path.join(datafolder, dataset_folder)
      dataset_path = parent_path

      # compute value of log_template_loc
      if args.dataset_collection == "M1":
        log_template_loc = "Failure\ Prediction\ Models/log_template_lists/M1_log_template_list.csv"
      elif args.dataset_collection == "M1":
        log_template_loc = "Failure\ Prediction\ Models/log_template_lists/M2_log_template_list.csv"
      elif args.dataset_collection == "M1":
        log_template_loc = "Failure\ Prediction\ Models/log_template_lists/M3_log_template_list.csv"
      else:
        print("data collection name is not correct, it should be either M1, M2, or M3")

      # compute name of results_name
      results_name = args.model +"_" +args.dataset_collection +".csv"

      #run the model for it
      command = "python Failure\ Prediction\ Models/main_non_DL.py "+ args.model +" "+ args.embedding_strategy +" "+ dataset_path+" " + log_template_loc
      command += " --result_file "+ args.results_loc
      os.system(command)
