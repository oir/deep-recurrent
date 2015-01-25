## Overview

Implementation of a deep bidirectional recurrent neural network for the task of opinion expression extraction.

See the paper,
>"Opinion Mining with Deep Recurrent Neural Networks"

>Ozan Irsoy, Claire Cardie

>EMNLP 2014

for details.

If you use my code, please cite:

	@InProceedings{irsoy-drnt,
	  author = {\.Irsoy, Ozan and Cardie, Claire},
	  title = {Opinion Mining with Deep Recurrent Neural Networks},
	  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
	  pages = {720--728},
	  year = {2014},
	  location = {Doha, Qatar},
	  url = {http://aclweb.org/anthology/D14-1080}
	}

Feel free to ask questions: oirsoy [a] cs [o] cornell [o] edu.
<http://www.cs.cornell.edu/~oirsoy/drnt.htm>

## Getting Started

Assuming you have g++ and the code here, running the bash script as

	bash run.sh

should

1. download small word embeddings (25 dimensional CW)
2. download the preprocessed MPQA dataset
3. download the Eigen library
4. compile and run to train a small model on the ESE task to be saved to disk.

That's it! Once you have a working setup, you can play with the hyperparameters or pick different word embeddings (300d word2vec is used in the experiments in the paper).

##License

Code is released under [the MIT license](http://opensource.org/licenses/MIT).