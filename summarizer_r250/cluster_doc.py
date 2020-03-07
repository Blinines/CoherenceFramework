# -*- coding: utf-8 -*-
import os
import re
import subprocess
import nltk
import tqdm
from pre_process import get_stem_words
from sentence import Sentence
from grid_model import main_grid_model
from hits_model import HITS, GRAM_ROLE_TO_SCORE_UNWEIGHTED

CONFIG_CLUSTER_SAVING = {
    'store_sentences_doc': True,
    'sentences_doc': './data/sentences_doc/',
    'sentences_trees': './data/sentences_trees/',
    'sentences_grid': './data/sentences_grid/'
}


class ClusterDoc(object):
    def __init__(self, texts_path, cluster_id, config_cluster):
        """
        @class ClusterDoc : base class for a set of documents based on cluster id.
        Creates if not exists tree and grid representation of the sentences.
        HITS algorithm can then be applied.
        @param texts_path : folder containing all the documents
        @param cluster_id : unique identifier for the cluster id
        @param config_cluster : cf example above. 
        Contains folder path to store the different representations
        """
        self.texts_path = texts_path
        self.cluster_id = cluster_id
        self.config_cluster = config_cluster
        self.porter = nltk.PorterStemmer()
        self.sentences_with_sharp, self.sent_object = self.get_sentences_doc(texts_path=texts_path)
        self.sentences = [elt for elt in self.sentences_with_sharp if elt != '#']

        self.text_path = '{0}{1}.txt'.format(
            config_cluster['sentences_doc'], cluster_id)
        self.tree_path = '{0}{1}.txt'.format(
            config_cluster['sentences_trees'], cluster_id)
        self.grid_path = '{0}{1}.txt'.format(
            config_cluster['sentences_grid'], cluster_id)

        if (not os.path.exists(self.text_path)) and \
            (config_cluster['store_sentences_doc']):
            self.save_sentences_text()

        if (not os.path.exists(self.tree_path)):
            self.save_sentences_trees()
        
        if (not os.path.exists(self.grid_path)):
            self.save_grid()

    def process_file(self, file_path):
        """ Extracting relevant information from the original DUC formatted data """
        with open(file_path, 'r') as f:
            text_init = f.read()
        
        # extract content in TEXT tag and remove tags
        text_clean = re.search(r"<TEXT>.*</TEXT>", text_init, re.DOTALL)
        if text_clean is None:
            return []
        text_clean = re.sub("<TEXT>\n", "", text_clean.group(0))
        text_clean = re.sub("\n</TEXT>", "", text_clean)

        # if article from LA, additionally removes P tags
        if 'LA' in file_path:
            text_clean = re.sub("<P>\n", "", text_clean)
            text_clean = re.sub("\n</P>", "", text_clean)

        # replace all types of quotations by normal quotes
        text_clean = re.sub("\n", " ", text_clean)
        text_clean = re.sub("\"", "\"", text_clean)
        text_clean = re.sub("''", "\"", text_clean)
        text_clean = re.sub("``", "\"", text_clean)	
        text_clean = re.sub(" +", " ", text_clean)
        text_clean = re.sub("1/2 ", "and a half ", text_clean)

        # segment data into a list of sentences
        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sentence_token.tokenize(text_clean.strip())	
        return lines

    def get_sentences_doc(self, texts_path):
        """ Storing into list all sentences of the set of documents.
        Raw sentence stored, as well as sentence objects """
        sent_raw = ['#']
        sent_object = []
        id_ = 0
        for file_path in os.listdir(texts_path):
            lines = self.process_file(file_path=texts_path+file_path)
            for line in lines:
                sent_raw.append(line)
                orig_words, stemmed_sent = get_stem_words(line=line,
                                                          porter=self.porter)
                sent_object.append(Sentence(doc_name=file_path,
                                            pre_pro_words=stemmed_sent,
                                            orig_words=orig_words,
                                            id_=id_))
                id_ += 1
            sent_raw.append('#')

        return sent_raw, sent_object
    
    def save_sentences_text(self):
        f = open(self.text_path, 'w+')
        index_doc = 0
        for sent in self.sentences_with_sharp:
            if sent == '#':
                if index_doc != 0:
                    f.write('\n')
                f.write('# clusterid={0} docid={1} \n'.format(self.cluster_id, index_doc))
                index_doc += 1
            else:
                f.write('{0}\n'.format(sent))
        f.close()
    
    def save_sentences_trees(self):
        command_line = "python parsedoctext.py -i {0} -o {1}".format(
            self.text_path, self.tree_path
        )
        print(command_line)
        subprocess.call(command_line, shell=True)
    
    def save_grid(self):
        command_line = "python grid_model.py -d {0} -s {1}".format(
            self.tree_path, self.grid_path
        )
        subprocess.call(command_line, shell=True)
    
    def apply_hits(self, config):
        self.hits = HITS(grid_path=self.grid_path, 
                         gram_role_to_score=config['gram_role_to_score'],
                         initial=config['initial'],
                         max_iter=config['max_iter'],
                         epsilon=config['epsilon'])
        self.entities = self.hits.entities
        self.sent_to_entities = self.hits.sent_to_entities
        self.hits.update_score()
        self.hits_rankings = self.hits.authority_score

        for index, ranking in enumerate(self.hits_rankings):
            self.sent_object[index].update_hits_score(score=ranking)
            self.sent_object[index].update_entities(ent=self.sent_to_entities[index])


if __name__ == '__main__':
    # Creating doc - tree - grid for each set of documents
    duc_folder_path = './duc2005/DUC2005_Summarization_Documents/duc2005_docs/'
    for i in tqdm.trange(len(os.listdir(duc_folder_path))):
        cluster_id = os.listdir(duc_folder_path)[i]
        
        # Initializing all clusters, i.e. saving all info we will need for later
        try:
            cluster = ClusterDoc(texts_path=duc_folder_path+cluster_id+'/',
                                cluster_id=cluster_id,
                                config_cluster=CONFIG_CLUSTER_SAVING)
        except Exception as e:
            errors = open('./errors_{0}.txt'.format(cluster_id), 'w')
            errors.write('{0}\n'.format(e))
            errors.close()
