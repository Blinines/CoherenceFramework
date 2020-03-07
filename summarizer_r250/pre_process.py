# -*- coding: utf-8 -*-
"""
Taken from 
https://github.com/vishnu45/NLP-Extractive-NEWS-summarization-using-MMR/blob/master/mmr_summarizer.py
"""
import re
import nltk


def get_stem_words(line, porter):
    # original words of the sentence before stemming
    orig_words = line[:]
    line = line.strip().lower()

    # word tokenization
    sent = nltk.word_tokenize(line)
    
    # stemming words
    stemmed_sent = [porter.stem(word) for word in sent]		
    stemmed_sent = filter(lambda x: x!='.'and x!='`'and x!=','and x!='?'and x!="'" 
        and x!='!' and x!='''"''' and x!="''" and x!="'s", stemmed_sent)
        
    return orig_words, stemmed_sent