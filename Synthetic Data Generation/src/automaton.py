"""
This file contains Automaton class which is used in data_generator.py and utils.py files.
In initialization, dot file of a DFA is read with the help of PySimpleAutomata library.

This Automaton class has more attributes than the original the automaton. The neighbor states
of each state is stored as neighbor_states attribute to facilitate the computations for the
random walk.

failure_pattern is a list of unique regular expressions that specifies the failure sequences
of the generated dataset
"""
from PySimpleAutomata import automata_IO  # to read the dot file of automata

from . import utils

MAX_INT = 1000000


# read the failure regular expressions from the input location
# and return the list of regular expressions
def read_regular_expressions(regular_expression_loc):
    with open(regular_expression_loc) as file:
        # each regular is in one line
        lines = [line.rstrip() for line in file]

    # convert each line to regular expression
    regular_expressions = set(r'{}'.format(l) for l in lines)
    return list(regular_expressions)


class Automaton:
    # creates the automaton and initial its the attributes
    def __init__(self, automaton_loc, failure_patterns_loc, min_length, max_length, size):

        self.automaton_loc = automaton_loc
        # read the dot file of input automata
        # the output is a dictionary of states, initial_state, accepting_states, and
        # transitions as the keys are of the dictionary
        automaton_dictionary = automata_IO.dfa_dot_importer(self.automaton_loc)

        self.states = automaton_dictionary['states']
        self.initial_state = automaton_dictionary['initial_state']
        self.accepting_states = automaton_dictionary['accepting_states']
        self.transitions = {}
        for key, value in automaton_dictionary['transitions'].items():
            if "\\n" in key[1]:
                log_tempaltes = key[1].split("\\n")
                for t in log_tempaltes:
                    self.transitions[(key[0], t)] = value

            elif "\n" in key[1]:
                log_tempaltes = key[1].split("\n ")
                for t in log_tempaltes:
                    self.transitions[(key[0], t)] = value

            else:
                self.transitions[key] = value
              
        # store the possible transitions from each state
        self.next_transitions = {}
        for state in self.states:
          self.next_transitions[state] = []
          for key, value in self.transitions.items():
            if key[0]== state:
              self.next_transitions[state].append((value, key[1]))

        # compute the neighbor and terminal states
        self.neighbor_states = self.find_neighbor_states()
        self.terminal_states = self.find_terminal_states()

        # read the regular expressions for failure pattern from the input location
        self.failure_patterns = read_regular_expressions(failure_patterns_loc)
        self.failure_patterns_n = len(self.failure_patterns)

        # store the minimum and maximum required length for sequences generated from this automaton
        self.min_length = min_length
        self.max_length = max_length
        # store all the failure sequences generated from failure patterns during initialization
        # the sequences are with respect to the max and min lengths
        self.generated_failure_sequences = {}
        for p in self.failure_patterns:
            failurelogsequences = utils.generate_failure_sequence(p, min_length, max_length, size)
            if len(failurelogsequences):
                self.generated_failure_sequences[p] = failurelogsequences

        # put the value of s
        self.s_values = self.calculate_s_values()

    # method to find the neighbor states of each state
    def find_neighbor_states(self):
        # dictionary of neighbor states that keys are the states and the value is
        # the set of all neighbors
        neighbor_states = {}

        for s in self.states:
            neighbor_states[s] = set()
            # check all the transitions to find neighbor states
            for key, value in self.transitions.items():
                # check is the beginning state of the transition match the state
                # and its not a transition to itself
                if key[0] == s and value != s:
                    neighbor_states[s].add(value)

        return neighbor_states

    # find terminal states in an automaton
    def find_terminal_states(self):
        # terminal states will be stored in a set
        terminal_states = set()

        for s in self.states:
            # check if a states has not neighbor state which cannot be itself
            if not self.neighbor_states[s]:
                # add the state to the set
                terminal_states.add(s)

        return terminal_states

    # calculate the value of s for each state of the automaton
    # value of s is the length of the shortest path from a state to an accepting state
    def calculate_s_values(self):
        # empty dictionary that maps each state to its s value
        s_value = {}

        # bfs to accepting states for each node
        for state in self.states:
            # value of flag switches to 1 whenever bfs finds an accepting state
            flag = 0
            # list of paths, starting from state
            path_list = [[state]]
            # index for the queue of bfs
            path_index = 0
            # To keep track of previously visited nodes
            previous_nodes = {state}

            if state in self.accepting_states:
                s_value[state] = 0
                continue

            # while there is an unchecked path in the bfs queue
            while path_index < len(path_list):
                current_path = path_list[path_index]
                last_node = current_path[-1]
                next_nodes = self.neighbor_states[last_node]

                # check if there is an accepting states among next nodes
                for acc in self.accepting_states:
                    if acc in next_nodes:
                        flag = 1
                        break

                # if accepting state has been found will break the while loop
                if flag:
                    s_value[state] = len(current_path)
                    break

                # Add new paths
                for next_node in next_nodes:
                    if not next_node in previous_nodes:
                        new_path = current_path[:]
                        new_path.append(next_node)
                        path_list.append(new_path)
                        # To avoid backtracking
                        previous_nodes.add(next_node)
                # Continue to next path in list
                path_index += 1

            if not flag:
                # No path is found
                # value of s will be a great number more than the range of mlsl
                s_value[state] = MAX_INT

        return s_value
