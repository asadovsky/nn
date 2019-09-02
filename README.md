# References

## TODO

- Develop baseline intent classifiers
  - BiLSTM with GloVe embeddings
- Develop baseline slot predictors
  - BOW over n^2 spans, softmax across all span labels
  - Use Viterbi algorithm during decoding
- Experiment
  - Character vs. word vs. WordPiece embeddings
  - FastText, ULMFiT, ELMo, BERT

## Semantic parsing

### Papers

- https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf
- https://www.aclweb.org/anthology/P17-1044
- https://arxiv.org/pdf/1611.01436.pdf

### Other

- https://github.com/yvchen/JointSLU
- https://github.com/luheng/deep_srl
- http://deeplearning.net/tutorial/rnnslu.html

## Embeddings

- https://nlp.stanford.edu/projects/glove/
- https://fasttext.cc/

## Formats

- https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
