# Synthetic Data Generation
This is folder contain the implementation of the Synthetic Data Generator to generate artifiical data sets. 


## Paper
- Fatemeh Hadadi, Joshua H. Dawes, Donghwan Shin, Domenico Bianculli, Lionel Briand, "[Systematic Evaluation of Deep Learning Models for Failure Prediction](https://...)", please use the following citation when you want to cite our work:
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

## Behavioral Model and Failure Patterns
A DFA dot file in which:

- by reading a log template, it transits to a state,

- each log template can appear in several transitions, and they are not specific to one transition,

- there are some terminal states

A failure pattern is a regular expression which is a subset the automaton that the generated sequences
from this failure pattern are considered as failure sequences.

Other generated sequences that are not accepted by a failure pattern, are labeled as normal.

## Run Synthesised Data Generator
```shell script
(venv) python main.py <automaton> <failure_patterns> <dataset_folder> [--failure_perc PAILUREPERC] [--dataset_name DATASETNAME] [--output_dir OUTPUTDIR] [--min_size MIINSIZE] [--max_size MAXSIZE] [--min_length MINLENGTH] [--max_length MAXLENGTH]

usage:

  automaton: the location of the automaton dot file
  failure_patterns: the location of desired failure patterns in terms of regular
  expression file in txt. The log template IDs are separated with ","
  
optional arguments:
  --failure_perc FAILUREPERC: the percentage of the failure sequence in generated dataset (default = 50) 
  --dataset_name DATASETNAME: the name of output dataset
  --output_dir OUTPUTDIR: the directory to save the output dataset in
  --min_size MINSIZE: the minimum number of sequences in dataset (default=100) 
  --max_size MAXSIZE: the maximum number of sequences in dataset (default=10000) 
  --min_length MINLENGTH: the minimum length of each sequence in dataset (default=1) 
  --max_length MAXLENGTH: the maximum length of each sequence in dataset (default=10000)
```

Examples:
```shell script
 (venv) python main.py Behavioral\ Model/M3/Linux.dot Behavioral\ Model/M3/failure_pattern_type1_Linux.txt --failure_perc 30
```

## Licensing
To be determined ...

