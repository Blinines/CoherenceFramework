# -*- coding: utf-8 -*-
import json
import argparse
import subprocess
from pyrouge import Rouge155

if __name__ == '__main__':
    # python rouge_score.py -t weighted -l 0.3
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=True, help="type of grid to build " + 
														"(weighted/unweighted)")
    ap.add_argument("-l", "--lambda", required=True, help="lambda value used ")
    args = vars(ap.parse_args())
    
    root_dir = '/home/ib431/Documents/projects/CoherenceFramework/summarizer_r250/'
    rouge_dir = root_dir + 'ROUGE-1.5.5'
    rouge_args = '-e' + root_dir + 'ROUGE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 250'
    rouge = Rouge155(rouge_dir, rouge_args)

    # 'model' refers to the human summaries 
    model_dir = root_dir + 'duc2005/results/ROUGE/models'
    model_filename_pattern = 'D#ID#.M.250.[A-Z].[A-Z]'

    print("-----------------MMR--------------------------")
    # 'system' or 'peer' refers to the system summaries
    system_dir = root_dir + 'data/summaries/{0}/{1}'.format(args["type"], args["lambda"])
    system_filename_pattern = 'd(\d+)'
    
    cmd_line = 'pyrouge_evaluate_plain_text_files ' + \
               '-s {0} '.format(system_dir) + \
               '-sfp "{0}" '.format(system_filename_pattern) + \
               '-m {0} '.format(model_dir) + \
               '-mfp {0}'.format(model_filename_pattern)
    subprocess.call(cmd_line, shell=True)