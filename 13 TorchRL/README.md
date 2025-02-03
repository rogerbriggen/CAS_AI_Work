# This shows how to solve cart-pole challenge from gymnasium (see https://gymnasium.farama.org/environments/classic_control/cart_pole/) with torchRL (https://pytorch.org/rl/stable/tutorials/getting-started-5.html) and solves the gymnasium challenge Acrobot https://gymnasium.farama.org/environments/classic_control/acrobot/ with torchRL

## Environment

The conda environment is named torchRL.

### The environment can be created with conda

````shell
conda env create -f environment.yml
````

### The environment can be updated with conda

````shell
conda env update --file environment.yml --prune
````

### To save the updated environment

````shell
conda env export > environment.yml
````

### Install a specific version

````shell
pip install --force-reinstall -v "numpy==1.26.4"
````

### Additional Software

- No additinal software needed

### Environment variables

No environment variables needed

## Start the optuna dashboard

````shell
optuna-dashboard sqlite:///noisy_dqn2.db
````
