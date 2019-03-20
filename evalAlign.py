# !/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
# from decode import *
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

The two references are quite different. The google translated reference contains
commas and apostrophes while the non-google reference does not. For obvious
reasons, the non-google translated references are more grammatical. For example,
"That is true for every member of Parliament."  is a lot better than "This
applies to all deputies.". The former is also more context specific, while the
latter looses some of the specificity; "deputies" is a lot more general than
"member of parliament". Using more correct references will always reflect the
ability of your model better than fewer. To improve the BLEU score, we should
perhaps add a parameter that takes into account the number of references and
scales the score accordingly.

These results are in one way as expected, and in another very unexpected. As
expected, with an increase in the n-gram size, we saw a decrease in accuracy.
This is because:
1. The likelihood of correctly translating three words, is a approximately 3x
lower than correctly translating one word. The average BLEU score on unigrams
for 1000 sentences was 0.28, while for bigrams it was 0.14. This is a 2x
reduction in size, as expected. This pattern remains constant even when trained
on more sentences. The trigrams have an average of 0, which is a bit unexpected,
this is probably due to (2).
2. Given that all three words have been correctly translated, the probability
that this correct translation is in on of the two reference sentences, is quite
low. If instead of two reference sentences we had a lot more, we could increase
the accuracy.

With an increase in training size, I expected there to be a significant increase
in the BLEU score. While there was an increase, it was not as large as expected.
For the unigrams, I saw a 0.02 increase in the average BLEU score from 1000
training sentences to 10000 training sentences. After 10000 training sentences
we saw a slight decrease in the average BLEU score for unigrams. This decrease
is perhaps just random noise, 25 testing sentences are quite few to a valid
performance measurement. Also with more training data, we see no more increase
in performance. The reason for this must be that after the first 1000 training
sentences, our estimated parameters were very close the the optimal parameters.
Adding a lot more data, will not shift them significantly thereafter. To
increase the BLEU score we'd have to fundamentally change the model, more data
will not make a difference.


"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model
    """

    if use_cached is True:
        LM = pickle.load(open(fn_LM + '.pickle', "rb"))
    else:
        LM = lm_train(data_dir, language, fn_LM)

    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model
    """
    if use_cached is True:
        AM = pickle.load(open(fn_AM + '.pickle', 'rb'))
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)

    return AM


def calculate_brevity(candidate, references):
    diff = None
    candidate_length = len(candidate.split(' '))
    reference_length = None
    for reference in references:
        ref_length = len(reference.split(' '))
        tmp_diff = math.fabs(candidate_length - ref_length)
        if diff is None or tmp_diff < diff:
            diff = tmp_diff
            reference_length = ref_length
    brevity_score = reference_length / candidate_length

    if brevity_score >= 1:
        brevity_score = math.exp(1 - brevity_score)
    else:
        brevity_score = 1

    return brevity_score


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """

    bleu_scores = []

    for index in range(len(eng)):
        r1 = eng[index]
        r2 = google_refs[index]
        c = eng_decoded[index]

        r = [r1, r2]
        bleu_score = 1
        for i in range(n):
            i = i + 1

            bleu_score *= BLEU_score(c, r, i)
        bleu_score = bleu_score ** (1 / n)
        bleu_score *= calculate_brevity(c, r)
        bleu_scores.append(bleu_score)

    return bleu_scores


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    LM = _getLM('u/cs401/A2 SMT/data/Hansard/Training/', 'e', 'LM', use_cached=True)

    sentence_lengths = [1000, 10000, 15000, 30000]
    bigram_sizes = [1, 2, 3]
    BLEU_scores = []

    task5_e_f = open('u/cs401/A2 SMT/data/Hansard/Testing/Task5.e', 'r')
    task5_f_f = open('u/cs401/A2 SMT/data/Hansard/Testing/Task5.f', 'r')
    task5_g_f = open('u/cs401/A2 SMT/data/Hansard/Testing/Task5.google.e', 'r')

    french = task5_f_f.readlines()
    french = [preprocess(french[i], 'f') for i in range(len(french))]

    eng = task5_e_f.readlines()
    eng = [preprocess(eng[i], 'f') for i in range(len(eng))]

    eng_g = task5_g_f.readlines()
    french = [preprocess(eng_g[i], 'f') for i in range(len(eng_g))]

    for sentence_length in sentence_lengths:
        AM = _getAM('u/cs401/A2 SMT/data/Hansard/Training/', sentence_length, 150, 'AM', use_cached=True)
        eng_decoded = [decode.decode(french_word, LM, AM) for french_word in french]
        for n in bigram_sizes:
            score = _get_BLEU_scores(eng_decoded, eng, eng_g, n)
            BLEU_scores.append(score)
            print('sentences: {}, bigram count: {}, average: {} scores:'.format(sentence_length, n, np.mean(score)))
            print(score)

    # Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    #
    # f = open("Task5_.txt", 'w+')
    # f.write(discussion)
    # f.write("\n\n")
    # f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")
    #
    # for i, AM in enumerate(AMs):
    #
    #     f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
    #     # Decode using AM #
    #     # Eval using 3 N-gram models #
    #     all_evals = []
    #     for n in range(1, 4):
    #         f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
    #         evals = _get_BLEU_scores(...)
    #         for v in evals:
    #             f.write(f"\t{v:1.4f}")
    #         all_evals.append(evals)
    #
    #     f.write("\n\n")
    #
    # f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    # f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()
    main('args')
