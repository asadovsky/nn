# nn

## Environment setup

### Requirements

- Python version 3.11 or above
- Node.js version 16 or above (for Pyright)

### pyenv

- https://github.com/pyenv/pyenv
- https://github.com/pyenv/pyenv-virtualenv

### venv

See [here](https://docs.python.org/3/library/venv.html) for background on venv.

We recommend adding this to your `~/.bashrc` file:

    export PIP_REQUIRE_VIRTUALENV=1
    alias vn='python3 -m venv .venv'
    alias va='source .venv/bin/activate'
    alias vd='deactivate'

### pip

Install required packages:

    $ pip install -r requirements.txt

Add a new package (using `numpy` as an example):

    $ echo numpy >> requirements.in && sort -u -o requirements.in requirements.in
    $ pip install -r requirements.in
    $ pip freeze > requirements.txt

## Training on Lambda Cloud

Launch an instance on https://lambdalabs.com/ and connect to it using SSH.

Install Python 3.11, create venv, and clone repo:

    $ sudo apt update
    $ sudo apt install python3.11 python3.11-venv
    $ python3.11 -m venv .venv
    $ git clone https://github.com/asadovsky/nn.git

Download a run:

    $ scp -r -i ~/.ssh/id_ed25519 user@ip-address:~/nn/.runs/run-id .

## Resources

### Code

- https://github.com/karpathy/build-nanogpt
- https://github.com/karpathy/nanoGPT
- https://github.com/karpathy/llm.c
- https://github.com/karpathy/minGPT
- https://github.com/karpathy/makemore

### Papers

- https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf
- https://arxiv.org/pdf/1611.01436.pdf
- https://aclweb.org/anthology/P17-1044/
- https://aclweb.org/anthology/N18-2050/
- https://aclweb.org/anthology/P18-2058/
- https://arxiv.org/pdf/1805.01052.pdf
- https://arxiv.org/pdf/1810.02245.pdf

### Datasets

- ATIS
  - https://github.com/yvchen/JointSLU
- TOP
  - https://aclanthology.org/D18-1300.pdf
  - https://fb.me/semanticparsingdialog
- TOPv2
  - https://arxiv.org/pdf/2010.03546v1.pdf
  - https://fb.me/TOPv2Dataset
- https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey

### Embeddings

- https://nlp.stanford.edu/projects/glove/
- https://fasttext.cc/

### Formats

- https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
