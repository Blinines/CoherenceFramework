# -*- coding: utf-8 -*-
"""
mmr_summarizer initially taken from 
https://github.com/Bharat123rox/HITS-and-PageRank-Algorithm
Then adapted for the purpose of our experiment
"""

import ast
import numpy as np
from math import sqrt
from collections import defaultdict
import argparse

GRAM_ROLE_TO_SCORE_UNWEIGHTED = {
    'S': 1, 'O': 1, 'X': 1, '-': 0
}
GRAM_ROLE_TO_SCORE_WEIGHTED = {
    'S': 3, 'O': 2, 'X': 1, '-': 0
}


class HITS():
    def __init__(self, grid_path, gram_role_to_score, initial,
                 max_iter, epsilon):
        """
        @class HITS : base class for HITS algorithm
        @param grid_path : path to the entity grid representation
        of the set of documents we want to rank
        @param gram_role_to_score : weights of grammatical roles
        to create the entity graph
        @param initial : value to initialize the authority scores.
        Either 1, 1/nb_sentences or 1/sqrt(nb_sentences).
        Cf. self.init_auth below to see the mapping
        @param max_iter : Maximum number of updates the algorithm computes
        @param epsilon : We consider that the algorithm has converged if 
        the L2 norm difference between the two update scores is lower than epsilon
        """
        self.init_auth = {1: (lambda x: 1),
                          2: (lambda x: 1./x),
                          3: (lambda x: 1./sqrt(x))}
        self.max_iter = max_iter
        self.epsilon = epsilon

        with open(grid_path, 'r') as f:
            info = f.readlines()
        
        self.gram_role_to_score = gram_role_to_score
        self.entities = ast.literal_eval(info[0][1:])
        self.nb_entities = len(self.entities)
        self.sentences = info[1:]
        self.nb_sentences = len(self.sentences)
        self.sent_to_entities = defaultdict(lambda: [])

        self.entity_graph = self.convert_grid_to_graph(self.entities,
                                                       self.sentences)
        self.authority_score, self.hub_score = \
            self.init_scores(initial_auth=initial)
        

    def convert_grid_to_graph(self, entities, sentences):
        entity_graph = np.zeros((len(entities), len(sentences)))

        for index_sent, sent_info in enumerate(sentences):
            for index_gram, gram_role in enumerate(sent_info.replace('\n', '')):
                # entity index_gram has gram_role in index_sent
                entity_graph[index_gram][index_sent] = \
                    self.gram_role_to_score[gram_role]
                if gram_role != '-':
                    self.sent_to_entities[index_sent].append(self.entities[index_gram])
        
        return entity_graph
    
    def init_scores(self, initial_auth):
        authority_score = np.array(
            [self.init_auth[initial_auth](self.nb_sentences)] * \
                self.nb_sentences)

        hub_score = np.dot(self.entity_graph, authority_score)
        authority_score = self.rescale(score=authority_score)
        hub_score = self.rescale(score=hub_score)
        return authority_score, hub_score

    def rescale(self, score):
        scale = sqrt(np.sum(score**2))
        return score / scale
    
    def update_score(self):
        for nb_iter in range(self.max_iter):
            past_auth = self.authority_score
            past_hub = self.hub_score

            # Update rule
            self.authority_score = np.dot(
                np.transpose(self.entity_graph), self.hub_score)
            self.hub_score = np.dot(
                self.entity_graph, self.authority_score)
            self.authority_score = self.rescale(self.authority_score)
            self.hub_score = self.rescale(self.hub_score)

            # Checking convergence
            if (sqrt(np.sum((past_auth - self.authority_score)**2)) < self.epsilon) and \
               (sqrt(np.sum((past_hub - self.hub_score)**2)) < self.epsilon):
               print(nb_iter)
               break


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--grid", required=True, help="path to grid representation")
    args = vars(ap.parse_args())
    path = args['grid']

    hits_weighted = HITS(grid_path=path, gram_role_to_score=GRAM_ROLE_TO_SCORE_WEIGHTED,
                        initial=3, max_iter=100000, epsilon=1e-6)
    hits_weighted.update_score()
    print(sorted(zip(hits_weighted.authority_score,
                    [i for i in range(hits_weighted.nb_sentences)]), reverse=True)[:3])

    hits_unweighted = HITS(grid_path=path, gram_role_to_score=GRAM_ROLE_TO_SCORE_UNWEIGHTED,
                        initial=3, max_iter=100000, epsilon=1e-6)
    hits_unweighted.update_score()
    print(sorted(zip(hits_unweighted.authority_score,
                    [i for i in range(hits_unweighted.nb_sentences)]), reverse=True)[:3])
