# -*- coding: utf-8 -*-
'''
Copied and modified from the original `python` folder of the repo. Grid representation.

Created on 12 Jan 2015
 
 Constructs an entity grid from a given file containing ptb trees. The file may be English, French or German.
 The entity grid uses the Stanford Parser to identify all nouns in the input text.
 For the English version it additionally determines the grammatical role played by that entity in each particular occurance. 
 The various options are set on the commandline, to ensure correct parser is set.
 
 @author Karin Sim

'''
from codecs import open
import argparse
import sys
import traceback
import logging
import gzip
from grid import r2i, i2r
import StanfordDependencies
import numpy as np
from discourse.doctext import iterdoctext, writedoctext
from discourse.util import smart_open, read_documents
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nouns = ['NNP', 'NP','NNS','NN','N','NE']
subject =[ 'csubj', 'csubjpass','subj','nsubj','nsubjpass']
object_= ["pobj","dobj","iobj"] 



def open_file(path):
    if path.endswith('.gz'):
        return gzip.open(path)
    else:
        return open(path)
        

# input is in form of ptb trees. 
def main_grid_model(args):
    """ Extract entities and construct grid """
    try:
        with open(args.directory, 'rb') as fi:
            with open(args.save_directory, 'w', encoding='utf-8') as fo:
                entities, nb_sent, entities_in_order = extract_grids(fi)
                grid = construct_grid(entities, nb_sent, entities_in_order)
                output_grid(grid, fo, entities_in_order) 
            logging.info('done: %s', args.directory)
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info()))) 



def extract_grids(fi):
    """ Identify entities from ptb trees for document. store in dictionary for grid construction. """
    entities = defaultdict(lambda : defaultdict(dict))
    entities_in_order = []
    nb_sent = 0
    for lines, attrs in iterdoctext(fi):
        logging.debug('document %s', attrs['docid'])
        print ' extract '+str(len(lines))+' lines'
        for line in lines:
            entities, entities_in_order = convert_tree(line, entities,
                                                       nb_sent, entities_in_order)
            nb_sent += 1
        
    return entities, nb_sent, entities_in_order     
                 
        
#from dependencies extract nouns with their grammatical dependencies in given sentence
def convert_tree(line, entities, idx, entities_in_order):
    if line == '(())':
        return entities, entities_in_order
        
    print ' convert_tree with '+line
    sd = StanfordDependencies.get_instance(backend='subprocess')
    
    #ex='(ROOT(S(NP (PRP$ My) (NN dog))(ADVP (RB also))(VP (VBZ likes)(S(VP (VBG eating)(NP (NN sausage)))))(. .)))'
    
    dependencies = sd.convert_tree(line, debug=True)
    
    for token in dependencies:
        print token
        if token.pos in nouns :
            print ' .. is a noun-'+token.pos
            grammatical_role = '-'
            if token.deprel in subject: 
                grammatical_role = 'S'
            elif token.deprel in object_:
                grammatical_role = 'O'
            else:
                grammatical_role = 'X'
            
            ''' if this entity has already occurred in the sentence, store the reference with highest grammatical role , 
            judged here  as S > O > X '''
            token_lemma = lemmatizer.lemmatize(token.form.lower())
            if token_lemma in entities and  entities[token_lemma][idx] :
                print str(entities[token_lemma][idx]) + ' comparing to '+str(r2i[grammatical_role])
                if (entities[token_lemma][idx]) < r2i[grammatical_role]:
                    entities[token_lemma][idx] = r2i[grammatical_role]
            else:
                entities[token_lemma][idx] = r2i[grammatical_role]
            
            if token_lemma not in entities_in_order:
                entities_in_order.append(token_lemma)
            ''' entity->list of : sentence_number->grammatical_role'''
    return entities, entities_in_order


def construct_grid(entities, sentences, entities_in_order):
    """ #construct grid from dictionary, rows are sentences, cols are entities """
    print 'size='+str(len(entities))
    grid = np.zeros((sentences, len(entities)))
    entity_idx = 0
    for entity in entities_in_order:
        occurances = entities[entity]
        for sentence in occurances :
            grid [sentence][entity_idx] = occurances[sentence] 
        entity_idx+=1
    return grid


def output_grid(grid, ostream, entities_in_order):
    """ output grid  """
    ostream.write('#{0}\n'.format(entities_in_order))
    for i, _ in enumerate( grid) :
        line = []
        for j, _ in enumerate (grid[i]): #each char representing entity
            line.append(i2r[int(grid[i][j])])
        ostream.write('{0}\n'.format(''.join(line)))
            

def parse_args():
    """parse command line arguments"""
    
    parser = argparse.ArgumentParser(description='implementation of Entity grid using ptb trees as input',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #parser.description = 'implementation of Entity grid'
    #parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    
    parser.add_argument('--directory', '-d', 
            type=str,
            #argparse.FileType('rb'),
            help="path for input file")
    parser.add_argument('--save_directory', '-s',
            type=str,
            help="path for saving the output file")
    
    #parser.add_argument('language', 
    #        type=argparse.FileType('r'),
    #        help="language of input file: one of English, French or German")
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')
    
    args = parser.parse_args()
    
    logging.basicConfig(
            level=(logging.DEBUG if args.verbose else logging.INFO), 
            format='%(levelname)s %(message)s')
    return args

if __name__ == '__main__':
    main_grid_model(parse_args())