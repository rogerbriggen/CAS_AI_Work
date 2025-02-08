# We now use Deep Reinforcment Learning - Hands-On, 3rd Edition, Lapan

see <https://leanpub.com/deepreinforcementlearninghands-on-thirdedition>
see <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition>, Chapter 11

## Environment

Python is a dependency hell... to run it use

````shell
conda create --name lapan
conda activate lapan
conda install -c conda-forge python=3.11
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt  # the requirements file is from https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/requirements.txt
````

Use <https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition> for the code.

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
