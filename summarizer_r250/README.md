# Multi-document Summarizer [R250 module]

-----------------------

Implementation of an unsupervised extractive multi-document summarizer, composed of three main parts :

* The bipartite graph representation of the set of documents

* The HITS ranking algorithm

* A modified version of the MMR-MD score

## Installation

* Clone this repo to your computer by using `git clone https://github.com/Blinines/CoherenceFramework.git`
* Follow the instructions on the original repo [README](../python/README.md) for Python
* Go in the main folder for our implementation : `cd summarizer_r250`
* When using scripts from the `summarizer_r250` folder, we assume you are in this specific folder for path.
* You will also need to download the stanford parser. We used the `stanford-parser-full-2014-10-31` version.
* You can get the DUC datasets from <https://www-nlpir.nist.gov/projects/duc/data.html>

## Usage

### Creating the summaries

The main script to run is the [mmr_summarizer](./mmr_summarizer.py), to create a summary for a particular set of documents. The system will first create a set of document object, `ClusterDoc`, which will create if non-existing the tree representation and the grid representation of the set of documents. To speed up the process, you can first generate all the grids by using the [cluster_doc](./cluster_doc.py) script. All the arguments for the mentioned scripts are detailed in the corresponding files.

### Evaluating with ROUGE automatic metric

We used `ROUGE-1.5.5` version (We used the ROUGE directory from <https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5>)
We then used the Python wrapper by <https://github.com/bheinzerling/pyrouge>
The script [rouge_score](./pyrouge/rouge_score.py) displays an example of how to get the ROUGE scores for all the summaries.  

## Describing project architecture

* [data]  : folder you should build to make the current scripts work, once you have the DUC dataset.
  * [sentences_doc] : Raw concatenated sentences from each set of documents
  * [sentences_grid]
  * [sentences_trees]
  * [summaries]
* [duc2005] : the datasets should be in the main folder
* [pyrouge](./pyrouge) : Python wrapper for ROUGE
* [ROUGE-1.5.5](./ROUGE-1.5.5) : original ROUGE Perl code
* [cluster_doc](./cluster_doc.py) : Representation of a set of documents. Creating a tree and a grid representation for each topic
* [grid_model](./grid_model.py) : Building grid representation
* [grid](./grid.py)
* [hits_model](./hits_model.py) : Implementation of the HITS algorithm
* [ldc](./ldc.py)
* [make_summaries](./make_summaries.py) : Command lines to generate all summaries for all topics, given a lambda value
* [mmr_summarizer](./mmr_summarizer.py) : Modified version of the MMR-MD algorithm
* [parsedoctext](./parsedoctext.py) : Parsing sentences
* [pre_process](./pre_process.py) : Word stemming
* [rouge_score](./rouge_score.py) : To be put in the pyrouge folder when you cloned the pyrouge repo.
* [sentence](./sentence.py) : Sentence object
