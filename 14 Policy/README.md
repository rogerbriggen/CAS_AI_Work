# We now use Deep Reinforcment Learning - Hands-On, 3rd Edition, Lapan

see <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition>, Chapter 11

## Environment

Python is a dependency hell... to run it use

````shell
conda create --name lapan
conda activate lapan
conda install -c conda-forge python=3.11
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
````

Use <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition/commit/83b6f971c853df12872b7fb13786e0237f34501e>

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

## Start the optuna dashboard

````shell
optuna-dashboard sqlite:///noisy_dqn2.db
````
