from preprocess import *
import pickle
import os


def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM

    INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language	: (string) either 'e' (English) or 'f' (French)
    fn_LM		: (string) the location to save the language model once trained

    OUTPUT

    LM			: (dictionary) a specialized language model

    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which
    incorporate unigram or bigram counts

    e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    language_model = dict()
    language_model['uni'] = dict()
    language_model['bi'] = dict()

    for file in os.listdir(data_dir):
        if file[-1] == language:
            data = open(data_dir + file, 'r')
            for line in data:
                line = preprocess(line, language).split(' ')
                for i in range(len(line)):

                    if line[i] not in language_model['uni']:
                        language_model['uni'][line[i]] = 1
                    else:
                        language_model['uni'][line[i]] += 1

                    if i != 0:
                        if line[i - 1] not in language_model['bi']:
                            language_model['bi'][line[i - 1]] = dict()
                            language_model['bi'][line[i - 1]][line[i]] = 1
                        else:
                            if line[i] not in language_model['bi'][line[i - 1]]:
                                language_model['bi'][line[i - 1]][line[i]] = 1
                            else:
                                language_model['bi'][line[i - 1]][line[i]] += 1

            data.close()

    # Save Model
    with open(fn_LM + '.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model
