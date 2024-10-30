**🚧 This repository is under development and not yet ready for use 🚧**

This is the official implementation of the paper "A Pre-Trained Graph-Based Model for AdaptiveSequencing of Educational Documents".

It contains several modules, mainly for the training of the recommender system, the simulation of the student population and the extraction of keywords from documents.

In order to run the experiments properly, please include the files (pre-trained models) provided in the supplementary materials at the right paths.


## Installation

To install the required dependencies, please run this in command line:
```
./install.sh
```

## Basic Usage

We use weights&biases for the logging of experimental results.

To pretrain the model on the source tasks, execute this command from the root folder:
```
python -m src.rl_pretrain
```

To finetune the model on the target task, execute this command from the root folder:
```
python -m src.rl_finetune
```

## Baselines

To run the Bassen et al. baseline, execute this command from the root folder:
```
python -m src.rl_finetune_bassen
```

To run the Vassoyan et al. baseline, execute this command from the root folder:
```
python -m src.rl_finetune_vassoyan
```

To run the CMAB baseline, execute this command from the root folder:
```
python -m src.baseline_irt.main
```

