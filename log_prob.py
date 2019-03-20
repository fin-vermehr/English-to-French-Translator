from preprocess import *
from lm_train import *
from math import log2

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing

    INPUTS:
    sentence :	(string) The PROCESSED sentence whose probability we wish to compute
    LM :		(dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta : 	(float) smoothing parameter where 0<delta<=1
    vocabSize :	(int) the number of words in the vocabulary

    OUTPUT:
    log_prob :	(float) log probability of sentence
    """
    sentence = sentence.split(' ')
    probabilities = []

    for i in range(1, len(sentence)):
        word = sentence[i]
        previous_word = sentence[i - 1]

        if smoothing is False:
            if previous_word in LM['uni']:
                uni_count = LM['uni'][previous_word]

                if word in LM['bi'][previous_word]:
                    bi_count = LM['bi'][previous_word][word]
                    probabilities.append(log2(bi_count / uni_count))
                else:
                    probabilities.append(float('-inf'))

            else:
                probabilities.append(float("-inf"))
                continue
        else:
            if previous_word in LM['uni']:
                uni_count = LM['uni'][previous_word]
                if word in LM['bi'][previous_word]:
                    bi_count = LM['bi'][previous_word][word]
                else:
                    bi_count = 0
            else:
                uni_count = 0

            probabilities.append(log2((bi_count + delta) / (uni_count + (delta * vocabSize))))
    log_prob = 0
    for prob in probabilities:
        log_prob += prob

    return log_prob
