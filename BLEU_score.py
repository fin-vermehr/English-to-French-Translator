import math

def BLEU_score(candidate, references, n, brevity=False):
	"""
	Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
	n=1 unigram
	n=2 bigram
	... and so on

	DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

	INPUTS:
	sentence :	(string) Candidate sentence. "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

	candidate_n = to_ngram(candidate, n)
	N = len(candidate_n)

	references_n = list()

	for reference in references:
		references_n.append(to_ngram(reference, n))

	C = 0
	for ngram in candidate_n:
		j = 0
		flag = False

		while j != len(references_n) and flag is False:
			if ngram in references_n[j]:
				C += 1
				flag = True
			else:
				j += 1

	bleu_score = (C / N)

	if brevity > 0:
		# TODO: add the 1/n?
		# bleu_score = bleu_score ** (1 / n)
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

		bleu_score *= brevity_score
	return bleu_score


def to_ngram(string, n):
	ngram_list = []
	words = string.split(' ')
	for j in range(n - 1, len(words)):
		ngram = []
		for i in range(n):
			ngram.append(words[j - (n - 1) + i])
		ngram_list.append(ngram)
	return(ngram_list)
