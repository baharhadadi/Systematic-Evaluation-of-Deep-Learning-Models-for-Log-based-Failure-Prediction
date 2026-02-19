# Failure Prediction Models (RQ1, RQ2, RQ3, RQ4, RQ5)

## Paper
- Fatemeh Hadadi, Joshua H. Dawes, Donghwan Shin, Domenico Bianculli, Lionel Briand, "Systematic Evaluation of Deep Learning Models for Failure Prediction", please use the following citation when you want to cite our work:
> ...


## Authors
- Fatemeh Hadadi (donghwan.shin@uni.lu)
- Joshua H. Dawes (joshua.dawes@uni.lu)
- Donghwan Shin (d.shin@sheffield.ac.uk)
- Domenico Bianculli (domenico.bianculli@uni.lu)
- Lionel Briand (lionel.briand@uni.lu, lbriand@uottawa.ca)


## Prerequisites
* python 3.8

## Install

Initialize python's virtual environment & install required packages:
```shell script
python3 -m venv venv
source venv/bin/activate 
(venv) pip install -r requirements.txt
```

## Run a Failure Prediction Model
```shell script
(venv) python main.py <dl_encoder> <embedding_strategy> <dataset_folder> <log_template_list> 
                      [--batch_size BATCHSIZE] [--epoch EPOCH] [--input_len INPUTLENGTH] [--results_loc RESULTS_LOCATION] 
                      [--results_name RESULTSANME] [--use_train_test]
usage:

  dl_encoder: options are LSTM, BiLSTM, CNN, and transformer
  embedding_strategy: options are BERT, Logkey2vec, and Fasttext+tf-idf
  dataset_folder: the folder that contains the dataset either raw or traina and test splitted
  log_template_list: the location of the log template list file that mapts log template id to their text
  
optional arguments:
  --batch_size: the size of batch size for training
  --epoch EPOCH: the numbe of eopchs for training
  --input_len INPUTLEN: the maximum length of the input for the dataset
  --result_name RESULTSNAME: the name of the results csv file
  --result_loc RESULTS_LOCATION: the location of results file (Default: \Results)
  --use_train_test: use the already splited data with train and test otherwise it uses raw folder
```

Example*:
```shell script
 (venv) python main.py CNN Logkey2vec Datasets/train_test/M2/1_100_5000_10 log_template_lists/M2_log_template_list.csv --batch_size 60 --epoch 20 --input_len 100 --result_name CNN_Logkey2vec_M2.csv --result_loc Results --use_train_test
```
*For the above example, copy the Datasets folder in this directory. 
