# This shows how to optimize with aputna the cart-pole challenge from gymnasium (see https://gymnasium.farama.org/environments/classic_control/cart_pole/) with the given example for Noisy Layer https://colab.research.google.com/drive/14JjQdSZpjKt3JWrslrJ9eEa1CL8CW9W7?usp=sharing and Dueling https://colab.research.google.com/drive/1RrEFdfilPwEaWfQOoxY8LOB70AxAp_l9?usp=sharing and solves the gymnasium challenge Acrobot https://gymnasium.farama.org/environments/classic_control/acrobot/

## Environment

The conda environment is named gym.

The environment can be created with conda:

````shell
conda env create -f environment.yml
````

To save the updated environment:

````shell
conda env export > environment.yml
````

### Additional Software

- No additinal software needed

### Environment variables

No environment variables needed

## Start the optuna dashboard

````shell
optuna-dashboard sqlite:///noisy_dqn2.db
````
