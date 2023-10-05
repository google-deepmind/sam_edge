## Running experiments

This code is provided to enable readers of ["Sharpness-Aware Minimization and the Edge of Stability"](https://arxiv.org/abs/2309.12488) to reproduce the experiments reported there, check implementation details, and run related experiments.

The libraries needed to run the code are listed in `requirements.txt` and a sample script to install the required libraries and run one quick experiment is provided in `run.sh`.

The two commands that run experiments are `image_classification` and `transformer`.  The default command-line arguments for `image_classification` are set to re-run an MNIST experiment.  For example, to examine how the operator norm of the Hessian compares with the SAM-edge when SAM's offset hyperparameter rho is 0.1, you can run
```
python3 -m image_classification --rho=0.1
```
then look at the graph in `/tmp/eigs_se_only.pdf`.  If you don't want to wait four hours, though, consider changing the `--time_limit_in_hours` flag, and possibly training a smaller network (see the list of command-line arguments below).

The image classification software uses TFDS to get the training data from the web on the fly.  To run a language modeling experiment, you need to download the [data](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) yourself. Once you do, and split it into training and test data, you can run, for example,
```
python3 -m transformer --rho=0.1 \
        --training_data=tinyshakespeare_train.txt \
        --test_data=tinyshakespeare_test.txt \
```

## Command-line arguments for image classification experiments

- rho (float): the distance between the location where SAM evaluates the gradient, and the current iterate
- time_limit_in_hours (float): limit on the amount of training time
- hessian_check_gap (float): the number of hours between evaluations of the hessian norm (and other statistics)
- step_size (float): learning rate
- batch_size (int): batch size
- nn_architecture (str): neural network architecture, either 'MLP' or 'CNN'
- dataset (str): either 'mnist' or 'cifar10'
- mlp_depth (int): depth of the network, if an MLP is used
- mlp_width (int): width of the network, if an MLP is used
- cnn_num_blocks (int): the number of blocks in the CNN architecture
- cnn_layers_per_block (int): the number of convolutional layers in each block
- cnn_feature_multiplier (int): the number of channels in the first convolutional layer
- mini_training_set_num_batches (int): if this is not None, make a reduced training set with this number of minibatches
- mini_test_set_num_batches (int): if this is not None, make a reduced test set with this number of minibatches
- eigs_curve_output (str): where to output the PDF file with plots of eigenvalues and edge-of-stability thresholds
- eigs_se_only_output (str): where to output the PDF file with plots of eigenvalues and the SAM-edge only
- alignment_curve_output (str): where to output the PDF file with alignments between gradients and the principal eigenvector of the Hessian
- loss_curve_output (str): where to output the PDF file giving the training error
- raw_data_output (str): where to output raw data, to potentially be used to generate ad-hoc plots
- num_principal_components (int): the number of principal eigenvalues of the Hessian to compute

## Command-line arguments for language modeling experiments

- training_data (str): the location of a text file with training data
- test_data (str): the location of a text file with test data
- rho (float): the distance between the location where SAM evaluates the gradient, and the current iterate
- time_limit_in_hours (float): limit on the amount of training time
- hessian_check_gap (float): the number of hours between evaluations of the hessian norm (and other statistics)
- step_size (float): learning rate
- batch_size (int): batch size
- eigs_curve_output (str): where to output the PDF file with plots of eigenvalues and edge-of-stability thresholds
- eigs_se_only_output (str): where to output the PDF file with plots of eigenvalues and the SAM-edge only
- alignment_curve_output (str): where to output the PDF file with alignments between gradients and the principal eigenvector of the Hessian
- loss_curve_output (str): where to output the PDF file giving the training error
- raw_data_output (str): where to output raw data, to potentially be used to generate ad-hoc plots
- num_principal_components (int): the number of principal eigenvalues of the Hessian to compute
- num_layers (int): the number of layers in the Transformer model
- num_heads (int): the number of attention heads in the Transformer model
- key_size (int): the key size of the Transformer model
- model_size (int): the ``model size'' of the Transformer model
- sequence_length (int): the sequence learning of the Transformer model

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
