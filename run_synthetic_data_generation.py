import argparse
import pandas as pd
import os

if __name__ == '__main__':
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('behavioralmodel', help="the location of the automaton that is going to be use for data generation",
                        type=str)
    parser.add_argument('--directory_name', help="The name of the directory which stores the datasets (default = GeneratedDatasets)",
                        type=str, default='GeneratedDatasets')
    args = parser.parse_args()

    # load the behavioral model and failure patterns
    if args.behavioralmodel == "M1_model":
      automaton = "Synthetic\ Data\ Generation/Behavioral\ Model/M1/NGLClient.dot"
      failure_patterns_type1 = "Synthetic\ Data\ Generation/Behavioral\ Model/M1/failure_pattern_type1_NGL.txt"
      failure_patterns_type2 = "Synthetic\ Data\ Generation/Behavioral\ Model/M1/failure_pattern_type2_NGL.txt"

    elif args.behavioralmodel == "M2_model":
      automaton = "Synthetic\ Data\ Generation/Behavioral\ Model/M2/HDFS.dot"
      failure_patterns_type1 = "Synthetic\ Data\ Generation/Behavioral\ Model/M2/failure_pattern_type1_HDFS.txt"
      failure_patterns_type2 = "Synthetic\ Data\ Generation/Behavioral\ Model/M2/failure_pattern_type2_HDFS.txt"

    elif args.behavioralmodel == "M3_model":
      automaton = "Synthetic\ Data\ Generation/Behavioral\ Model/M3/Linux.dot"
      failure_patterns_type1 = "Synthetic\ Data\ Generation/Behavioral\ Model/M3/failure_pattern_type1_Linux.txt"
      failure_patterns_type2 = "Synthetic\ Data\ Generation/Behavioral\ Model/M3/failure_pattern_type2_Linux.txt"

    else:
      print("the behavioral model chosen does not exist here")

    # create the data set directory
    output_dir  = args.directory_name.replace("\\","")
    output_dir = output_dir.replace("  ", " ")
    
    if not os.path.exists(args.directory_name):
      os.mkdir(output_dir)
    

    # read the combinations
    df = pd.read_csv("DataSampleCombinations.csv")

    # run the data generator
    for index, row in df.iterrows():
      FPtype = "F" if int(row[1][-1])==1 else "I"
      # print the data set information that is going to be generated
      print("{} dataset: type: {} MLSL: {} DS: {} FP: {}".format(index, FPtype,row[2],row[3],row[4]))
      
      # if the data sample is from type 1 of failure patterns, the type 1 failure pattern list would be use
      if row[1] == "type 1":
        
        # create the name of the data set and its directory name
        dataset_name = "1"+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4])+".csv"
        parent_directory = "1"+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4])
        parent_path = os.path.join(output_dir, parent_directory)

        # if the directory storing the data set already exist
        if os.path.exists(parent_path):
            # make the command to run data generator main file
            path = parent_path
            path = path.replace(" ", "\ ")
            command = "python Synthetic\ Data\ Generation/main.py {} {} --max_length {} --min_size {} --max_size {} --failure_perc {} --dataset_name {} --output_dir {}".format(automaton, failure_patterns_type1, row[2], row[3], row[3], row[4], dataset_name, path)
            os.system(command)
            continue
        # if the directory does not exist, create it and then generate the data set
        os.mkdir(parent_path)
        path = parent_path
        path = path.replace(" ", "\ ")
        command = "python Synthetic\ Data\ Generation/main.py {} {} --max_length {} --min_size {} --max_size {} --failure_perc {} --dataset_name {} --output_dir {}".format(automaton, failure_patterns_type1, row[2], row[3], row[3], row[4], dataset_name, path)
        os.system(command)  

      # similiar to type 1, type 2 is generated
      elif row[1] == "type 2":
                
        # create the name of the data set and its directory name
        dataset_name = "2"+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4])+".csv"
        parent_directory = "2"+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4])
        parent_path = os.path.join(output_dir, parent_directory)

        # if the directory storing the data set already exist
        if os.path.exists(parent_path):
            path = parent_path
            path = path.replace(" ", "\ ")
            # make the command to run data generator main file
            command = "python Synthetic\ Data\ Generation/main.py {} {} --max_length {} --min_size {} --max_size {} --failure_perc {} --dataset_name {} --output_dir {}".format(automaton, failure_patterns_type2, row[2], row[3], row[3], row[4], dataset_name, path)
            os.system(command)
            continue
        # if the directory does not exist, create it and then generate the data set
        os.mkdir(parent_path)
        path = parent_path
        path = path.replace(" ", "\ ")
        command = "python Synthetic\ Data\ Generation/main.py {} {} --max_length {} --min_size {} --max_size {} --failure_perc {} --dataset_name {} --output_dir {}".format(automaton, failure_patterns_type2, row[2], row[3], row[3], row[4], dataset_name, path)
        os.system(command) 
