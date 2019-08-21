# References

## Next steps

- Evaluate on test set
- Plot accuracy and loss curves
- TensorBoard

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
- https://arxiv.org/pdf/1902.10909.pdf

### Other

- https://github.com/yvchen/JointSLU
- http://deeplearning.net/tutorial/rnnslu.html

## Embeddings

- https://nlp.stanford.edu/projects/glove/

## Formats

- https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
