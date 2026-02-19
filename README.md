# Systematic-Evaluation-of-Deep-Learning-Models-for-Failure-Prediction

This repository contains the **replication package** for the paper:

**"Systematic Evaluation of Deep Learning Models for Log-based Failure Prediction"**

If you use this replication package, please cite:

```bibtex
@article{Hadadi_2024,
   title={Systematic Evaluation of Deep Learning Models for Log-based Failure Prediction},
   volume={29},
   ISSN={1573-7616},
   url={http://dx.doi.org/10.1007/s10664-024-10501-4},
   DOI={10.1007/s10664-024-10501-4},
   number={5},
   journal={Empirical Software Engineering},
   publisher={Springer Science and Business Media LLC},
   author={Hadadi, Fatemeh and Dawes, Joshua H. and Shin, Donghwan and Bianculli, Domenico and Briand, Lionel},
   year={2024},
   month=jun
}
```
## Authors
- Fatemeh Hadadi (donghwan.shin@uni.lu)
- Joshua H. Dawes (joshua.dawes@uni.lu)
- Donghwan Shin (d.shin@sheffield.ac.uk)
- Domenico Bianculli (domenico.bianculli@uni.lu)
- Lionel Briand (lionel.briand@uni.lu, lbriand@uottawa.ca)

## Prerequisite
- Python 3 (python3.8.10 or higher is recommended)

Please initialize Python's virtual environment and install requirements:
```shell script
 python3 -m venv venv
 source venv/bin/activate
 (venv) pip install -r Synthetic\ Data\ Generation/requirements.txt
 (venv) pip install --no-deps --no-cache-dir --force-reinstall exrex
 (venv) pip install -r Failure\ Prediction\ Models/requirements.txt
```

- Dataset Availability

Due to GitHub size limitations, the datasets used in this study are hosted on Figshare and can be downloaded from:

https://doi.org/10.6084/m9.figshare.22219111

After downloading, place the datasets in the appropriate local dataset directory as described below.

## Running Synthentic Data Generation
Run the Synthetic Data Generator with any of the three behavioral models:
```shell script
 (venv) python run_synthetic_data_generation.py <behavioral_model_name> [--directory_name DIRECTORY_NAME]
usage: 
behavioral_model_name: options are M1_model, M2_model, and M3_model

optional argument: 
 --directory_name DIRECTORY_NAME: the name of the directory which stores the generated data sets (default = GeneratedDatasets)
 ``` 

Examples:
```shell script
 (venv) python run_synthetic_data_generation.py M1_model  --directory_name M1
 (venv) python run_synthetic_data_generation.py M2_model  --directory_name M2
 (venv) python run_synthetic_data_generation.py M3_model  --directory_name M3
```

## Running Failure Prediction Models (RQ1, RQ2, RQ3, RQ4, RQ5)
Run the script:
```shell script
 (venv) python run_DL_failure_predictors.py <dl_encoder> <embedding_strategy> 
                                               <dataset_collection> [--results_loc RESULTS_LOCATION] [--use_train_test]
usage:

  dl_encoder: options are LSTM, BiLSTM, CNN, and transformer
  embedding_strategy: options are BERT, and Logkey2vec, Fasttext+tf-idf
  dataset_collection: options are M1, M2, M3, OpenStack_FP
  
optional arguments:
  --results_loc RESULTS_LOCATION: the location of results file (Default: \Rep_Results)
  --use_train_test: use the already split data with train and test otherwise it uses the raw folder
  ```

Examples:
```shell script
 (venv) python run_DL_failure_predictors.py CNN Logkey2vec M1 --use_train_test
 (venv) python run_DL_failure_predictors.py BiLSTM BERT M2 --use_train_test
 (venv) python run_DL_failure_predictors.py Transformer Logkey2vec M3 --use_train_test
 (venv) python run_DL_failure_predictors.py CNN Logkey2vec OpenStack_FP --use_train_test
```
If you want to use your replicated data, put your folders of the synthisised generation data of "M1", "M2", and "M3" in the \Dataset\raw folder.

for RQ4, run the traditional ML-based failure predictor as follows:
```shell script
 (venv) python run_non_DL_failure_predictors.py <model> 
                                               <dataset_collection> [--results_loc RESULTS_LOCATION] [--use_train_test]
usage:

  model: option is RF
  dataset_collection: options are M1, M2, M3
  
optional arguments:
  --results_loc RESULTS_LOCATION: the location of the results file (Default: \Rep_Results)
  --use_train_test: use the already split data with train and test otherwise it uses the raw folder
  ```

Examples:
```shell script
 (venv) python run_non_DL_failure_predictors.py RF M1 --use_train_test
 (venv) python run_non_DL_failure_predictors.py RF M2 --use_train_test
 (venv) python run_non_DL_failure_predictors.py RF M3 --use_train_test
```
## Result Data Analysis
Install the requirements:
```shell script
 (venv) pip install -r data_analysis_requirements.txt 
```

Run the script:
```shell script
 (venv) python run_analyze_results.py 
```
In case of using your replication results, in the Rep_Results folder as default, copy them to the \Results folder which should have 24 (4x2x3 experiments) csv files.


The scripts will automatically generate the following files:
- `rq1-boxplot.pdf`
- `rq2-boxplot-combined.pdf`
- `rq2-boxplot-cnn.pdf`
- `rq3-cnn-rf.pdf`
- `rq4-cnn-l-a.pdf`
- `rq4-cnn-l-b.pdf`
- `rq4-cnn-l-c.pdf`
- `rq4-cnn-l-d.pdf`
- `rq4-bilstm-b-b.pdf`
- `rq4-decisiontree.pdf`
- `rq4-cnn-l-regressiontree.pdf`
- `rq4-cnn-b-regressiontree.pdf`
- `rq4-bilstm-b-regressiontree.pdf`

