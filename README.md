# Description

Implementation of [SIF](https://github.com/PrincetonML/SIF) sentence embedding in Java.

* Use pre-computed tfidf weights & word2vec to compute weighted sentence embedding.
* Use [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to compute principle components for all pairs of sentences and remove it.

For usage see test cases.

# Data

Please download word2vec to data. I used glove.6B.50d.txt from [Glove](https://nlp.stanford.edu/projects/glove/).
