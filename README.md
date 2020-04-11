# AETF: Autoencoding Topographic Factors

This code provides a reference implementation of the AETF algorithm described in the following publications:

  * [Autoencoding Topographic Factors](https://www.liebertpub.com/doi/full/10.1089/cmb.2018.0176). \
  Moretti, A.\*, Atkinson-Stirn, A.\*, Marks, G.\*, Pe'er, I. \
  Journal of Computational Biology, 2019 26(6):546–560. PMID: 30526005

  * [Auto-Encoding Topographic Factors](https://www.cs.columbia.edu/~amoretti/papers/AETF.pdf). \
  Moretti, A.\*, Atkinson-Stirn, A.\*, Pe'er, I. \
  Joint IJCAI-ICML Workshop on Computational Biology, 2018.
  
AETF is an extension of the TFA algorithm from the following publication. As a reference, TFA is also supported in the repo.
  
  * [Topographic Factor Analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094914). \
  Manning, J., Raganath, R., Norman, K., Blei, D. \
  Plos One, 2014
  
  ## Installation
  
  The code is written in Python 3.6. It should also run in Python 2.7 with the following dependencies:

* Tensorflow
* numpy
* scipy
* matplotlib

To checkout, run `git@github.com:amoretti86/AETF.git`
  
  
  ## Demo

| Original | Inferred |
|:--------------------------:|:--------------------------:|
| <img src="https://github.com/amoretti86/AETF/blob/master/figs/raw.png" width="300"/> | <img src="https://github.com/amoretti86/AETF/blob/master/Flow%20evolution%20across%20epochs.gif" width="350"/> 


