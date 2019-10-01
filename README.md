# References

## TODO

- Eval infra
  - https://github.com/chakki-works/seqeval
- Develop baselines for slot prediction
  - Use Viterbi algorithm when decoding IOB tag predictions
    - https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_decode
  - Predict labels for n^2 spans
- Experiment
  - Joint (or pipelined) prediction of intent and slots
  - Character vs. word vs. WordPiece embeddings
  - FastText, ULMFiT, ELMo, BERT

## Semantic parsing

### Papers

- https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf
- https://arxiv.org/pdf/1611.01436.pdf
- https://aclweb.org/anthology/P17-1044
- https://aclweb.org/anthology/P18-2058
- https://arxiv.org/pdf/1805.01052.pdf
- https://arxiv.org/pdf/1810.02245.pdf

### Other

- https://github.com/yvchen/JointSLU
- https://github.com/luheng/deep_srl
- http://deeplearning.net/tutorial/rnnslu.html

## Embeddings

- https://nlp.stanford.edu/projects/glove/
- https://fasttext.cc/

## Formats

- https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
