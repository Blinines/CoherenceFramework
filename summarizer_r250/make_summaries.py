# -*- coding: utf-8 -*-
import subprocess
from os import listdir
from datetime import datetime

# Values of lambda used : 0.7 - 0.5 - 0.3

lamb = 0.3

if __name__ == '__main__':
    """ Creates weighted and unweighted summaries for value lamb for all the topic documents.
    Necessitates to have the duc2005 data in the corresponding folder """
    doc_ids = listdir('./duc2005/DUC2005_Summarization_Documents/duc2005_docs/')

    for docid in doc_ids:
        date_begin = datetime.now()
        print('Process began for {0}\t {1}'.format(docid, date_begin))
        cmd_line = 'python mmr_summarizer.py -c {0} -t {1} -i 1 -iter 100000 -e 1e-6 -l {2} -w1 1 -w2 1 -wa 1 -wb 1' 
        subprocess.call(cmd_line.format(docid, 'weighted', lamb), shell=True)
        date_end_w = datetime.now()
        print('Weighted summary : done at \t {0}, took \t {1}'.format(date_end_w, date_end_w-date_begin))
        subprocess.call(cmd_line.format(docid, 'unweighted', lamb), shell=True)
        date_end_u = datetime.now()
        print('Unweighted summary : done at \t {0}, took \t {1}'.format(date_end_u, date_end_u-date_end_w))
        print('Finished creating summaries for \t {0}, took \t {1}'.format(docid, date_end_u-date_begin))
        print('==========')