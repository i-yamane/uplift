# uplift
Code for uplift modeling.

## Requirements:
* python >=3
* numpy
* scipy
* chainer ==3
* scikit-learn
* matplotlib

## Installation:
* Git clone this repository:
```bash
git clone https://github.com/i-yamane/uplift.git
```
* Install as a library:
```bash
pip install uplift
```
* For development/debugging purposes, you may want to do this instead:
```bash
pip install -e uplift
```

## Update:
* Pull the latest changes and upgrade with pip:
```bash
cd /path/to/uplift
git pull
pip install --upgrade .
```

## Modules:
* with_joint_labels: This module contains methods for uplift modeling from joint labels.
* with_separate_labels: This module contains methods for [uplift modeling from separate labels](https://arxiv.org/abs/1803.05112 "Uplift Modeling from Separate Labels").

