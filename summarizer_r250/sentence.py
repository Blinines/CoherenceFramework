# -*- coding: utf-8 -*-
from collections import defaultdict
"""
mmr_summarizer initially taken from 
https://github.com/vishnu45/NLP-Extractive-NEWS-summarization-using-MMR/blob/master/mmr_summarizer.py
Then adapted for the purpose of our experiment
"""
#----------------------------------------------------------------------------------
# Description:	Sentence class to store setences from the individual files in the
#				document cluster.
#----------------------------------------------------------------------------------

from nltk.corpus import stopwords

def sent_word_freq(pre_pro_words):
    """Word frequencies of sentence object"""
    word_freq = defaultdict(int)
    for word in pre_pro_words:
        word_freq[word] += 1
    return word_freq


class Sentence():

    def __init__(self, doc_name, pre_pro_words, orig_words,
                 id_, entities=None, hits_ranking=None):
        self.doc_name = doc_name
        self.pre_pro_words = pre_pro_words
        self.orig_words = orig_words
        self.id_ = id_
        self.hits_ranking = hits_ranking
        self.word_freq = sent_word_freq(pre_pro_words)
        self.entities = entities
    
    def update_hits_score(self, score):
        self.hits_ranking = score
    
    def update_entities(self, ent):
        self.entities = ent
	