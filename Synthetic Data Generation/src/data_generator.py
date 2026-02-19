"""
This file contains the Data Generator object. This has run(), save() and output_statistics() method to facilitate the
 generation, storage and reporting of the result.

 imported files:
 automaton file is imported here to create the automaton object from input dot file and use it for the guided
 random walk.
 utils file contains functions, such as guided_random_walk() and stop_condition that are used in data
 generation phase in run() method of the DataGenerator class.
"""
from . import automaton
from . import utils
import random
import csv
import os
import numpy as np


# main object for data generation
class DataGenerator:
    # initialized the dataset and automaton
    def __init__(self, automaton_loc, failure_patterns, min_length, max_length, size):
        self.automaton_loc = automaton_loc
        # dataset is a list of tuples that each tuple has two elements: sequences of log
        # template IDs and the label as failure/normal
        self.dataset = []
        # the automaton takes shape from input dot file
        #  if failure patterns are specified as input argument
        self.automaton = automaton.Automaton(self.automaton_loc, failure_patterns, min_length, max_length, size)

        self.min_length = min_length
        self.max_length = max_length
        return

    # main method for data generation with respect to desired percentage of failure sequences and
    # maximum and minimum size of dataset
    def run(self, failure_perc, min_size, max_size):

        # mask to check all the failure states are visited at least once during data generation
        mask = {s: 0 for s in self.automaton.generated_failure_sequences.keys()}
        failure_pattern_options = [p for p in self.automaton.generated_failure_sequences.keys()]

        # guided random walk  loop
        while not utils.stop_condition(self.dataset, failure_perc, mask, min_size, max_size):

            r = random.random()  # random variable for sequence generation mode in a way to keep the failure percentage

            if r >= (failure_perc / 100):
                # generate normal sequence
                sequence, _ = utils.filtered_random_walk(self.automaton, self.max_length)

                # check the length of the generated sequence
                while len(sequence) < self.min_length or len(sequence) > self.max_length or \
                        utils.is_failure_sequence(sequence, self.automaton):
                    # print("log sequence is not valid or normal:", sequence)
                    sequence, _ = utils.filtered_random_walk(self.automaton, self.max_length)

                self.dataset.append((sequence, 'normal'))

            else:
                # generate failure sequence
                # randomly choose a failure pattern to generate
                failure_pattern = random.choice(failure_pattern_options)
                # choose a sequence from generated failure sequences that match chosen failure pattern
                sequence = random.choice(self.automaton.generated_failure_sequences[failure_pattern])

                # check if this sequence is also valid in the automaton
                # it means that the sequence can be generated from random walk in automaton
                #while not utils.is_valid_sequence(self.automaton, sequence):
                    # choose another sequence
                    #print("failure sequence is not valid:", sequence)
                    #sequence = random.choice(self.automaton.generated_failure_sequences[failure_pattern])

                # add sequence to dataset
                self.dataset.append((sequence, 'failure'))

                # marks it as visited when the value > 0
                mask[failure_pattern] += 1

        return

    # save the dataset as csv file
    def save(self, output_dir, dataset_name):
        # create the file location
        file_loc = os.path.join(output_dir, dataset_name)
        # write the dictionary to a csv file
        with open(file_loc, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # first row is the header of csv
            writer.writerow(['sequence', 'label'])
            for sequence, label in self.dataset:
                writer.writerow([sequence, label])
        return

    # print the statistics report of the generated dataset
    def output_statistics(self):
        dataset_size = len(self.dataset)
        # keep the length of each sequence in a sequence_length list
        sequence_length = [len([n.strip() for n in sequence]) for sequence, _ in self.dataset]
        max_sequence_length = max(sequence_length)
        min_sequence_length = min(sequence_length)
        average_sequence_length = sum(sequence_length) / dataset_size
        sd_sequence_length = np.round(np.sqrt(np.var(sequence_length)))

        # the total number of failure labels in dataset
        failure_num = sum([1 for _, label in self.dataset if label == 'failure'])
        # the total number of normal labels in dataset
        normal_num = sum([1 for _, label in self.dataset if label == 'normal'])

        failure_percentage = (failure_num / dataset_size) * 100
        normal_percentage = (normal_num / dataset_size) * 100
        print('dataset statistics----------------------------------')
        print('dataset size: ', dataset_size, '\nsequence length: ', '\n\t average:', average_sequence_length)
        print('\t standard derivation: ', sd_sequence_length)
        print('\t minimum: ', min_sequence_length, '\tmaximum: ', max_sequence_length)
        print('failure sequence percentage: ', failure_percentage, '%\nnormal sequence percentage: ', normal_percentage,
              '%')
        return
