from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm.
    We assume that we are implemented P(foreign|english)

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter : 		(int) the maximum number of iterations of the EM algorithm
    fn_AM : 		(string) the location to save the alignment model

    OUTPUT:
    AM :			(dictionary) alignment model structure

    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
    is the computed expectation that the foreign_word is produced by english_word.

            LM['house']['maison'] = 0.5
    """
    AM = {}

    # Read training data
    data = read_hansard(train_dir, num_sentences)

    AM = initialize(data[0], data[1])
    print(len(data[0]))

    for i in range(0, max_iter):
        AM = em_step(AM, data[0], data[1])

    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Iterate between E and M steps
    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider


    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

    Make sure to read the files in an aligned manner.
    """
    i = 0

    sentence_list = []
    english_list = []
    french_list = []

    while i != num_sentences:
        for file in os.listdir(train_dir):
            if file[-1] == 'e' and 'Task5' not in file:
                english_file = open(train_dir + file, 'r')
                english_lines = english_file.read().splitlines()
                french_file = open(train_dir + file[:-1] + 'f', 'r')
                french_lines = french_file.read().splitlines()
                j = 0
                while i != num_sentences and j < len(french_lines):
                    english_list.append(preprocess(english_lines[j], 'e').split(' '))
                    french_list.append(preprocess(french_lines[j], 'f').split(' '))
                    j += 1
                    i += 1
        english_file.close()
        french_file.close()
    sentence_list.append(english_list)
    sentence_list.append(french_list)

    return sentence_list


def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    AM = dict()

    mapping_dict = dict()

    for i in range(len(eng)):
        for english_word in eng[i]:
            if english_word not in mapping_dict:
                mapping_dict[english_word] = set([])
            for french_word in fre[i]:
                mapping_dict[english_word].update(fre[i])

    for english_word in mapping_dict:
        AM[english_word] = dict()
        mapping_size = len(mapping_dict[english_word])

        for french_word in mapping_dict[english_word]:
            if french_word == 'SENTSTART' and english_word == 'SENTSTART':
                AM[english_word][french_word] = 1
            elif french_word == 'SENTEND' and english_word == 'SENTEND':
                AM[english_word][french_word] = 1
            else:
                AM[english_word][french_word] = 1 / (mapping_size - 2)

    return AM

def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    t_count = dict()
    total = dict()

    for index in range(len(eng)):
        for english_word in eng[index]:
            if english_word not in t_count:
                t_count[english_word] = dict()

            for french_word in fre[index]:
                t_count[english_word][french_word] = 0
        for english_word in eng[index]:
            total[english_word] = 0

    for index in range(len(fre)):
        for french_word in fre[index]:
            denom_c = 0
            f_count = fre[index].count(french_word)
            for english_word in eng[index]:
                denom_c += t[english_word][french_word] * f_count
            for english_word in eng[index]:
                e_count = eng[index].count(english_word)
                t_count[english_word][french_word] += t[english_word][french_word] * f_count * e_count / denom_c
                total[english_word] += t[english_word][french_word] * f_count * e_count / denom_c
    for english_word in total:
        for french_word in t_count[english_word]:
            t[english_word][french_word] = t_count[english_word][french_word] / total[english_word]
    return t
