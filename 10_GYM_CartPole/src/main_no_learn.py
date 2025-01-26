# see https://gymnasium.farama.org/environments/classic_control/cart_pole/ for more information
# This will solve the CartPole-v1 environment but not by learning... but by using a formula to determine the action to take.


import gymnasium as gym
from math import pow, fabs
from enum import Enum

class Action(Enum):
    LEFT = 0
    RIGHT = 1


def sgn(x):
    """
    Returns the sign of a number.

    Parameters:
    x (float or int): The number to check.

    Returns:
    int: -1 if x is negative, 1 if x is positive or zero.
    """
    return (-1 if x<0 else 1)

def action1(observation):
    """
    Determines the action to take based on the given observation.
    The action is calculated using the formula:
    x = sgn(a) * (1000 * |a|)^2 + sgn(p) * (100 * |p|)^2 + sgn(v) * (100 * |v|)^2
    where:
    - a is the pole angle
    - p is the cart position
    - v is the pole velocity
    Args:
        observation (tuple): A tuple containing four elements:
            - cart_position (float): The position of the cart.
            - cart_velocity (float): The velocity of the cart.
            - pole_angle (float): The angle of the pole.
            - pole_velocity (float): The velocity of the pole.
    Returns:
        int: The action to take. Returns 0 if the calculated value is less than 0, otherwise returns 1.
    """

    # x = sgn a (1000 a)² + sgn p (100 p)² + sgn v (100 v)²
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    a = pow(1000 * fabs(pole_angle), 2) * sgn(pole_angle)
    b = pow(100 * fabs(cart_position), 2) * sgn(cart_position)
    c = pow(100 * fabs(pole_velocity), 2) * sgn(pole_velocity)
    print(observation)
    ret = Action.LEFT.value if (a + b + c) < 0 else Action.RIGHT.value
    print(ret)
    return ret
    

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")


observation, info = env.reset(seed=42)
total_rewards = 0
print(observation)
for t in range(1000):
    env.render()
    # action = env.action_space.sample()
    # observation, reward, terminated, truncated, info = env.step(action)
    observation, reward, terminated, truncated, info = env.step(action1(observation))
    total_rewards += reward
    if terminated or truncated:
        print(f"Episode finished after {t+1} timesteps with reward {total_rewards}")
        print(f"observation: {observation}")
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"info: {info}")
        observation, info = env.reset()
        break
env.close()