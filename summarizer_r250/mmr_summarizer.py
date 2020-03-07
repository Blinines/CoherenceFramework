# -*- coding: utf-8 -*-
"""
mmr_summarizer initially taken from 
https://github.com/vishnu45/NLP-Extractive-NEWS-summarization-using-MMR/blob/master/mmr_summarizer.py
Then adapted for the purpose of our experiment
"""

import nltk
import os
import math
import string
import re
import argparse
import numpy as np
from sentence import Sentence
from nltk.corpus import stopwords
from collections import defaultdict
from cluster_doc import CONFIG_CLUSTER_SAVING, ClusterDoc
from hits_model import GRAM_ROLE_TO_SCORE_WEIGHTED, GRAM_ROLE_TO_SCORE_UNWEIGHTED


def tf(sentences):
    """ Term frequencies of the words within the document cluster """
    tfs = defaultdict(int)
    for sent in sentences:
	    for word in sent.word_freq.keys():
                tfs[word] += 1
    return tfs


def idf(sentences):
    """Inverse document frequencies for the words in the document cluster"""
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []
    
    for sent in sentences:
        for word in sent.pre_pro_words:

            # not to calculate a word's IDF value more than once
            if sent.word_freq.get(word, 0) != 0:
                words[word] = words.get(word, 0)+ 1

    for word in words:
        n = words[word]
        try:  # avoid zero division errors
            w2.append(n)
            idf = math.log10(float(N)/n)
        except ZeroDivisionError:
            idf = 0
        
        idfs[word] = idf
            
    return idfs


def tf_idf(sentences):
    """ TF-IDF score of the words within document cluster"""
    tfs = tf(sentences)
    idfs = idf(sentences)
    retval = defaultdict(lambda: [])

    for word in tfs:
        tf_idfs=  tfs[word] * idfs[word]
        retval[tf_idfs].append(word)

    return retval


def sentence_sim(sentence1, sentence2, idf_w):
    """ Sentence similarity using cosine similarity
    sentence1, first sentence
    sentence2, second sentence to which first sentence has to be compared
    """
    numerator = 0
    denominator = 0	
	
    for word in sentence2.pre_pro_words:		
		numerator+= sentence1.word_freq.get(word,0) * sentence2.word_freq.get(word,0) *  idf_w.get(word,0) ** 2

    for word in sentence1.pre_pro_words:
		denominator+= ( sentence1.word_freq.get(word,0) * idf_w.get(word,0) ) ** 2

    # check for divide by zero cases and return back minimal similarity
    try:
		return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
		return float("-inf")	

#---------------------------------------------------------------------------------
# Description	: Function to build a query of n words on the basis of TF-IDF value
# Parameters	: sentences, sentences of the document cluster
#				  IDF_w, IDF values of the words
#				  n, desired length of query (number of words in query)
# Return 		: query sentence consisting of best n words
#---------------------------------------------------------------------------------
def build_query(sentences, tf_idf_w, n):
	scores = tf_idf_w.keys()
	scores.sort(reverse=True)

	i, j = 0, 0
	query_words = []

	while i < n:  # selecting top n queries 
		words = tf_idf_w[scores[j]]
		for word in words:
			query_words.append(word)
			i += 1
			if i > n:
				break
		j += 1
	return Sentence(doc_name='query', pre_pro_words=query_words,
				    orig_words=query_words, id_='query')


#---------------------------------------------------------------------------------
# Description	: Function to create the summary set of a desired number of words 
# Parameters	: sentences, sentences of the document cluster
#				  best_sentnece, best sentence in the document cluster
#				  query, reference query for the document cluster
#				  summary_length, desired number of words for the summary
#				  labmta, lambda value of the MMR score calculation formula
#				  IDF, IDF value of words in the document cluster 
# Return 		: name 
#---------------------------------------------------------------------------------
def make_summary(sentences, query, summary_length,
                 lamb, idf, w1, w2, wa, wb):	
	summary = []
	sum_len = 0

	# keeping adding sentences until number of words exceeds summary length
	while (sum_len < summary_length):	
		mmr_val = {}		

		for sent in sentences:
			mmr_val[sent] = mmr_score(sent, query, summary, lamb, idf,
									  w1, w2, wa, wb)
		
		maxxer = max(mmr_val, key=mmr_val.get)
	 	summary.append(maxxer)
		sentences.remove(maxxer)
		sum_len += len(maxxer.pre_pro_words)
	
	return summary

#---------------------------------------------------------------------------------
# Description	: Function to calculate the modified MMR score given a sentence, the query
#				  and the current best set of sentences
# Parameters	: Si, particular sentence for which the modified MMR score has to be calculated
#				  query, query sentence for the particualr document cluster
#				  Sj, the best sentences that are already selected
#				  lamb, lambda value in the modified MMR formula
#				  IDF, IDF value for words in the cluster
# Return 		: name 
#---------------------------------------------------------------------------------
def mmr_score(s_i, query, s_j, lamb, idf,
			  w1, w2, wa, wb):	

    # Part for standard relevance
	if s_i.hits_ranking is not None:
		sim_1 = w1 * sentence_sim(s_i, query, idf) + w2 * s_i.hits_ranking
	else:
		sim_1 = sentence_sim(s_i, query, idf)
	l_expr = lamb * sim_1

    # If summary empty => only counting this part
	if len(s_j) == 0:
		return l_expr

    # Diveristy part
	value = [float("-inf")]
	for sent in s_j:
		sim_2 = wa * sentence_sim(s_i, sent, idf)
		if (s_i.entities is not None) and (sent.entities is not None):
			sim_2 += wb * len(set(s_i.entities).intersection(set(sent.entities)))
        value.append(sim_2)

	r_expr = (1-lamb) * max(value)

	return float(l_expr) - float(r_expr)	


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__=='__main__':	
	# python mmr_summarizer.py -c d301i -t unweighted -i 1 -iter 100000 -e 1e-6 -l 0.7 -w1 1 -w2 1 -wa 1 -wb 1 
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--cluster", required=True, help="cluster id (all files within " + 
														   "corresponding folder will be summarized)")
	ap.add_argument("-t", "--type", required=True, help="type of grid to build " + 
														"(weighted/unweighted)")
	ap.add_argument("-i", "--initial", required=True, help="initialization of scores for HITS ")
	ap.add_argument("-iter", "--iter", required=True, help="max iteration for HITS")
	ap.add_argument("-e", "--epsilon", required=True, help="epsilon value for HITS ")
	ap.add_argument("-l", "--lamb", required=True, help="lamb value for MMR score ")
	ap.add_argument("-w1", "--w1", required=True, help="w1 weight for MMR score ")
	ap.add_argument("-w2", "--w2", required=True, help="w2 weight for MMR score ")
	ap.add_argument("-wa", "--wa", required=True, help="wa weight for MMR score ")
	ap.add_argument("-wb", "--wb", required=True, help="wb weight for MMR score ")
	args = vars(ap.parse_args())

	main_folder_path = './duc2005/DUC2005_Summarization_Documents/duc2005_docs/'
	main_folder_save = './data/summaries/'

	cluster = ClusterDoc(texts_path=main_folder_path+args['cluster']+'/',
						 cluster_id=args['cluster'],
						 config_cluster=CONFIG_CLUSTER_SAVING)
	
	GRAM_ROLE_TO_SCORE = {
		'weighted': GRAM_ROLE_TO_SCORE_WEIGHTED,
		'unweighted': GRAM_ROLE_TO_SCORE_UNWEIGHTED
	}
	CONFIG_HITS = {
		'type': args['type'],
		'gram_role_to_score': GRAM_ROLE_TO_SCORE[args['type']],
		'initial': int(args["initial"]),
		'max_iter': int(args["iter"]),
		'epsilon': float(args["epsilon"])
	}

	cluster.apply_hits(CONFIG_HITS)
	sentences = cluster.sent_object

	# calculate TF, IDF and TF-IDF scores
	idf_w 		= idf(sentences)
	tf_idf_w 	= tf_idf(sentences)	

	# build query; set the number of words to include in our query
	query = build_query(sentences, tf_idf_w, 10)	

	# build summary by adding more relevant sentences
	summary = make_summary(sentences=sentences, query=query,
						   summary_length=250, lamb=float(args["lamb"]),
						   idf=idf_w, w1=int(args["w1"]), w2=int(args["w2"]),
						   wa=int(args["wa"]), wb=int(args["wb"]))
	
	final_summary = ""
	for sent in summary:
		final_summary = final_summary + sent.orig_words + "\n"
	final_summary = final_summary[:-1]
	name_save = '{0}{1}/{2}/{3}'.format(
		main_folder_save, args['type'], args['lamb'], args['cluster']
	)
	with open(name_save, 'w') as outfile:
		outfile.write(final_summary)