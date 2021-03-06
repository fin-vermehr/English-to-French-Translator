import re


def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language	: (string) either 'e' (English) or 'f' (French)
                  Language of in_sentence

    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    out_sentence = in_sentence.lower()
    out_sentence = re.sub('([\.\,\!\?\[\]\(\)\:\;\+\-\<\>\=\"])', r' \1 ', out_sentence)
    out_sentence = re.sub('(\'\s)', r' \1 ', out_sentence)
    out_sentence = re.sub(r'\s+', ' ', out_sentence).strip()
    out_sentence = 'SENTSTART ' + out_sentence + ' SENTEND'

    if language == 'f':
        out_sentence = re.sub('(\s+l\')', r'\1 ', out_sentence)
        out_sentence = re.sub('([bcfhjklmnpqrstvxz]\')(?=([a-z]))', r'\1 ', out_sentence)
        out_sentence = re.sub('([d]\')(?!(abord|accord|ailleurs|habitude))(?=([a-z]))', r'\1 ', out_sentence)
        out_sentence = re.sub('(\s+qu\')', r'\1 ', out_sentence)
        out_sentence = re.sub('(\')(?=(on|il))', r'\1 ', out_sentence)

        out_sentence = re.sub('(\s+l’)', r'\1 ', out_sentence)
        out_sentence = re.sub('([bcfhjklmnpqrstvxz]’)(?=([a-z]))', r'\1 ', out_sentence)
        out_sentence = re.sub('(\s+qu’)', r'\1 ', out_sentence)
        out_sentence = re.sub('(’)(?=(on|il))', r'\1 ', out_sentence)

    out_sentence = re.sub(r'\s+', ' ', out_sentence).strip()

    return out_sentence
