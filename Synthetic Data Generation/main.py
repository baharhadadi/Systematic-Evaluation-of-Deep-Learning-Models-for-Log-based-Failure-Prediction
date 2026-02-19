"""
This file contains the main part of the automata-based data generator.
Input parameters are as follows:
automaton: automaton location
--failure_patterns: The location of desired failure patterns in terms of regular
expression file in txt. The log template IDs are separated with ","
--failure_perc: percentage of the failure sequences in dataset (optional)
--dataset_name: the name of output dataset (optional)
--output_dir: the directory to save the output dataset in (optional)
--min_size: the minimum number of sequences in dataset (default=100) (optional)
--max_size: the maximum number of sequences in dataset (default=10000) (optional)
--min_length: the minimum length of each sequence in dataset (default=1) (optional)
--max_length: the maximum length of each sequence in dataset (default=10000) (optional)

After reading the parameters, data generator object will be created and dataset
is going to be generated with run() method. The output sequences are
saved in a csv file with their labels. At the end the statistics of the output dataset.
"""

import argparse
from src.data_generator import DataGenerator


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('automaton', help="the location of the automaton that is going to be use for data generation",
                        type=str)
    parser.add_argument('failure_patterns', help="The location of desired failure patterns in terms of regular "
                                                 "expression file in txt. The log template IDs are separated"
                                                 "with ,",
                        type=str)
    parser.add_argument('--failure_perc', help="The desired percentage of failure sequences in dataset (default = 50)",
                        type=int, default=50)
    parser.add_argument('--dataset_name', help="The name of output dataset ending in .csv (default = dataset.csv)",
                        type=str, default='dataset.csv')
    parser.add_argument('--output_dir', help="The directory for the output dataset",
                        type=str, default='')
    parser.add_argument('--min_size', help='The minimum number of sequences in dataset (default=100)',
                        type=int, default=100)
    parser.add_argument('--max_size', help='The maximum number of sequences in dataset (default=10000)',
                        type=int, default=10000)
    parser.add_argument('--min_length', help='The minimum length of each sequence in dataset (default=1)',
                        type=int, default=1)
    parser.add_argument('--max_length', help='The maximum length of each sequence in dataset (default=10000)',
                        type=int, default=10000)
    args = parser.parse_args()

    # initialize the data generator object with input arguments
    failure_size = int(args.max_size * args.failure_perc/100)
    data_generator = DataGenerator(args.automaton, args.failure_patterns, args.min_length, args.max_length, failure_size)

    # run the automata-based data generation
    data_generator.run(args.failure_perc, args.min_size, args.max_size)

    # save the output sequences of the data generator to the input directory
    data_generator.save(args.output_dir, args.dataset_name)

    # print the statistics report of the dataset such as the average length of the sequences.
    data_generator.output_statistics()



