# Day 14 - Policy Gradient Networks (PGN)

We now use the book Deep Reinforcment Learning - Hands-On, 3rd Edition, 2024, Lapan

see <https://leanpub.com/deepreinforcementlearninghands-on-thirdedition>
see <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition>, Chapter 11

My own copy with some changes of the excercises code is here: <https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition>

## Environment

Python is a dependency hell... to run it use

````shell
conda create --name lapan
conda activate lapan
conda install -c conda-forge python=3.11
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt  # the requirements file is from https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/requirements.txt
````

Use <https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/tree/main/Chapter11> for the code.

<https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter11/05_pong_pg_tune.py> does not work because of dependencies to tune... we would need to find the correct old version of tune... yay, dependency management in pyhton really sucks.

The conda environment is named lapan.

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

## Run

- Change to your local copy of the excercise
- Change to Chapter11
- Just run the different python files

## See results in tensorboard

````shell
# - Make sure you are in the correct conda env
# - Make sure you are in the Deep-Reinforcement-Learning-Hands-On-Third-Edition root directory
tensorboard --logdir=runs
````

Open webbrowser at <http://localhost:6006/> (or check the output of the tensorboard start)

## Start the optuna dashboard

````shell
optuna-dashboard sqlite:///11_04_lunarlander_pg.db
````
