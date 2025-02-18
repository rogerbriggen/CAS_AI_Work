# Day 15 - A2C (Advantage Actor Critic)

We now use the book Deep Reinforcment Learning - Hands-On, 3rd Edition, 2024, Lapan

see <https://leanpub.com/deepreinforcementlearninghands-on-thirdedition>
see <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition>, Chapter 12

My own copy with some changes of the excercises code is here: <https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition>

## Excercise

Get to run [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) with PG and A2C and compare them.

### Created files

- [Chapter11/04_lunarlander_pg.py](https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter11/04_lunarlander_pg.py) Copy of 04_cartpole_pg.py but adapted to LunarLander.
- [Chapter11/04_lunarlander_pg_optuna.py](https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter11/04_lunarlander_pg_optuna.py) More advanced versions with optuna to optimize hyper parameters. Changed mean reward to 150, added maxium episodes and added clipping for the gradients.
- [Chapter12/02_lunarlander_a2c.py](https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter12/02_lunarlander_a2c.py) Base version is 02pong_a2c.py but adapted for Lunar Lander.
- [Chapter12/02_lunarlander_a2c.py](https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter12/02_lunarlander_a2c.py) More advanced versions with optuna to optimize hyper parameters. Changed mean reward to 150

## Environment

Python is a dependency hell... if just importing enviorment.yml does not work, check aout Day 14 or the book instructions.

Use <https://github.com/rogerbriggen/Deep-Reinforcement-Learning-Hands-On-Third-Edition/tree/main/Chapter11> for the code.

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
- Change to Chapter12
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
optuna-dashboard sqlite:///12_02_lunarlander_a2c.db
````

## Results

Both versions were optimized in several optuna runs to find out good hyper parameters.
First with 20 trials and find good parameters and then with more episodes to find the results by using the 3 best paramters from the first round.

Unfortunately I cannot upload the runs and database since each run generates 1 GB of data for tensorboard and optuna database is smaller but optuna cannot show the values for the long running learning sessions...

### LunarLander PG

There was always the problem, that after a very good start the rewards dropped dramatically. With hyper parameter optimiziation and added clipping of the gradients and longer running training this problem was solved.




### LunarLander A2C

