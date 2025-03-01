# This shows how to solve LunarLander challenge from gymnasium (see <https://gymnasium.farama.org/environments/box2d/lunar_lander/>) with torchRL (<https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html>)

Code is from <https://github.com/pytorch/rl/blob/d4f88460abf0239f39c5677dca6f735a011697d9/tutorials/sphinx-tutorials/coding_ppo.py>

## Environment

The conda environment is named torchrl2.

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

## See results in tensorboard

````shell
# - Make sure you are in the correct conda env
# - Make sure you are in root directory
tensorboard --logdir=runs
````

Open webbrowser at <http://localhost:6006/> (or check the output of the tensorboard start)
