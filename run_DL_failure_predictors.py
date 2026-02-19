import argparse
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dl_encoder', help="the DL encoder type either CNN, BiLSTM, LSTM, or transformer",
                        type=str)
    parser.add_argument('embedding_strategy', help="the embedding strategy either BERT or Logkey2vec",
                        type=str)
    parser.add_argument('dataset_collection', help="the name of dataset collections either M1, M2, M3, or OpenStack",
                        type=str)                    
    parser.add_argument('--results_loc', help="the location of the folder storing results file",
                        type=str, default = "\Rep_Results")
    parser.add_argument('--use_train_test', help="whether to use splited data or raw data, default is raw",
                        dest='use_train_test', action='store_true', default=False)
    args = parser.parse_args()

    # read the level combination
    combination = "DataSampleCombinations.csv"
    df = pd.read_csv(combination, header = None)
    
    # set to location of dataset which is either from raw folder or train_test folder
    datasetfolder = "Datasets/raw/" + args.dataset_collection
    
    if args.use_train_test:
      datafolder = "Datasets/train_test/" + args.dataset_collection

    if args.dataset_collection != "OpenStack_FP":

      dataset_path = datafolder

      log_template_loc = "Failure\ Prediction\ Models/log_template_lists/OpenStack_log_template_list.csv"
      results_name = args.dl_encoder +"_" + args.embedding_strategy + "_" +args.dataset_collection +".csv"
      batch_size = 5
      epoch = 10

      # set to use whether the splited data or the raw data
      use_trian_test = ""
      if args.use_train_test:
        use_train_test = " --use_train_test"

      #run the model for it
      command = "python Failure\ Prediction\ Models/main_DL.py "+ args.dl_encoder +" "+ args.embedding_strategy +" "+ dataset_path+" " + log_template_loc
      command += " --batch_size {} --epoch {} --input_len {}".format(batch_size,epoch,500)
      command += " --result_name "+ results_name + " --result_loc "+ args.results_loc + use_train_test
      os.system(command)

    else:
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
          print("data collection name is not correct, it should be either M1, M2, M3, or OpenStack")

        # compute name of results_name
        results_name = args.dl_encoder +"_" + args.embedding_strategy + "_" +args.dataset_collection +".csv"

        # set the hyperparameters of the model
        if int(row[3]) == 200:
          epoch = 20
          batch_size = 10
          if row[4] <= 30:
              batch_size = 10
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5

        if int(row[3]) == 500:
          epoch = 20
          batch_size = 15
          if row[4] <= 30:
              batch_size = 15
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5
          
        if int(row[3]) == 1000:
          epoch = 20
          batch_size = 20
          if row[4] <= 30:
              batch_size = 30
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5
              epoch = 10

        if int(row[3]) == 5000:
          epoch = 20
          batch_size = 30
          if row[4] <= 30:
              batch_size = 60
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5
              epoch = 10

        if int(row[3]) == 10000:
          epoch = 20
          batch_size = 150
          if row[4] <= 30:
              batch_size = 300
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5
              epoch = 5

        if int(row[3]) == 50000:
          epoch = 20
          batch_size = 300
          if row[4] <= 30:
              batch_size = 600
          if int(row[2]) ==500 or int(row[2])==1000:
              batch_size = 5
              epoch = 5
      
        # set to use whether the splited data or the raw data
        use_trian_test = ""
        if args.use_train_test:
          use_train_test = " --use_train_test"

        #run the model for it
        command = "python Failure\ Prediction\ Models/main_DL.py "+ args.dl_encoder +" "+ args.embedding_strategy +" "+ dataset_path+" " + log_template_loc
        command += " --batch_size {} --epoch {} --input_len {}".format(batch_size,epoch,row[2])
        command += " --result_name "+ results_name + " --result_loc "+ args.results_loc + use_train_test
        os.system(command)
