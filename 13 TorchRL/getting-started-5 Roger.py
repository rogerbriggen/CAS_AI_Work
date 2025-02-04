"""
Get started with your own first training loop
=============================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_first_training:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl

"""

#################################
# Time to wrap up everything we've learned so far in this Getting Started
# series!
#
# In this tutorial, we will be writing the most basic training loop there is
# using only components we have presented in the previous lessons.
#
# We'll be using DQN with a CartPole environment as a prototypical example.
#
# We will be voluntarily keeping the verbosity to its minimum, only linking
# each section to the related tutorial.
#
# Building the environment
# ------------------------
#
# We'll be using a gym environment with a :class:`~torchrl.envs.transforms.StepCounter`
# transform. If you need a refresher, check our these features are presented in
# :ref:`the environment tutorial <gs_env_ted>`.
#

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

#torch.manual_seed(0)

import time
import numpy as np

from torchrl.envs import GymEnv, StepCounter, TransformedEnv

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")

#game_name = "CartPole-v1"
game_name = "Acrobot-v1"
print(f"Game name: {game_name}")
env = TransformedEnv(GymEnv(game_name), StepCounter())   
env.set_seed(0)

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

#################################
# Designing a policy
# ------------------
#
# The next step is to build our policy.
# We'll be making a regular, deterministic
# version of the actor to be used within the
# :ref:`loss module <gs_optim>` and during
# :ref:`evaluation <gs_logging>`.
# Next, we will augment it with an exploration module
# for :ref:`inference <gs_storage>`.

from torchrl.modules import EGreedyModule, MLP, QValueModule

n_observations = env.reset()

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, linear_features=512, linear_features_out=512, advantage_features=512, value_features=512):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, linear_features)
        self.layer2 = nn.Linear(linear_features, linear_features_out)

        self.fc_adv = nn.Sequential(
            nn.Linear(linear_features_out, advantage_features),
            nn.ReLU(),
            nn.Linear(advantage_features, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(linear_features_out, value_features),
            nn.ReLU(),
            nn.Linear(value_features, 1)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean() # ist gleich Q Value
if game_name == "CartPole-v1":    
    value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64], device=device)
elif game_name == "Acrobot-v1":
    value_mlp = DQN(n_observations=env.observation_spec['observation'].shape[-1], n_actions=env.action_spec.shape[-1], linear_features=2**10, linear_features_out=2**7, advantage_features=2**9, value_features=2**7).to(device)

value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"]).to(device)
policy = Seq(value_net, QValueModule(spec=env.action_spec)).to(device)
if isinstance(value_mlp, DQN):
    eps_init = 0.99
    eps_end = 0.01
    annealing_num_steps = 100_000  # Keep exploration high for a while
else:
    eps_init = 0.5
    eps_end = 0.1
    annealing_num_steps = 100_000
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=annealing_num_steps, eps_init=eps_init, eps_end=eps_end
).to(device)
policy_explore = Seq(policy, exploration_module).to(device)


#################################
# Data Collector and replay buffer
# --------------------------------
#
# Here comes the data part: we need a
# :ref:`data collector <gs_storage_collector>` to easily get batches of data
# and a :ref:`replay buffer <gs_storage_rb>` to store that data for training.
#

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer


init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
    device=device
)

if game_name == "CartPole-v1":
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
elif game_name == "Acrobot-v1":
    rb = TensorDictPrioritizedReplayBuffer(storage=LazyTensorStorage(100_000), alpha=0.7, beta=0.4, reduction=min) #, reduction=min because we want to minimize the amount of steps

from torch.optim import Adam

#################################
# Loss module and optimizer
# -------------------------
#
# We build our loss as indicated in the :ref:`dedicated tutorial <gs_optim>`, with
# its optimizer and target parameter updater:

from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
if game_name == "CartPole-v1":
    optim = Adam(loss.parameters(), lr=0.02)
elif game_name == "Acrobot-v1":
    optim = Adam(loss.parameters(), lr=1e-4)
    #optim = opt.AdamW(policy.parameters(), lr=1e-4, amsgrad=True)   

updater = SoftUpdate(loss, eps=0.99)


#################################
# Logger
# ------
#
# We'll be using a CSV logger to log our results, and save rendered videos.
#

from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder

path = "./training_loop"
logger = CSVLogger(exp_name="dqn", log_dir=path, video_format="mp4")
video_recorder = VideoRecorder(logger, tag="video")
record_env = TransformedEnv(
    #GymEnv("CartPole-v1", from_pixels=True, pixels_only=False), video_recorder
    GymEnv(game_name, from_pixels=True, pixels_only=False).to(device), video_recorder
).to(device)


#################################
# Inference (before training loop for faster iterations)
# -------------
def my_inference_callback(env, tensor_dict):
    # Example: Print the current step and reward
    print(f"Callback:\n  env: {env} \n  tensor_dict: {tensor_dict}")
    # You can add more custom logic here

def inference(policy, load_model=True):
    from tensordict.tensordict import TensorDict

    # ----- Inference -----
    if load_model:
        # If you want to run inference in a new session, reload the model state like:
        model_save_path = os.path.join("./training_loop", "policy_model.pt")
        policy.load_state_dict(torch.load(model_save_path))
    else:
        # Set the policy to evaluation mode and load the saved parameters.
        policy.eval()

    trajectory = env.rollout(max_steps=1000, policy=policy, callback=my_inference_callback)
    
    torchrl_logger.info(f"Inference rollout step count: {trajectory['step_count'][-1]}")

# Uncomment the following line to run just inference
#inference(policy=policy)
#sys.exit(0)

#################################
# Training loop
# -------------
#
# Instead of fixing a specific number of iterations to run, we will keep on
# training the network until it reaches a certain performance (arbitrarily
# defined as 200 steps in the environment -- with CartPole, success is defined
# as having longer trajectories).
#

total_count = 0
total_episodes = 0
t0 = time.time()

if game_name == "Acrobot-v1":
    last_indice = -1
    max_length = 500
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    if game_name == "CartPole-v1":
        max_length = rb[:]["next", "step_count"].max()
    if game_name == "Acrobot-v1":
        # We can either count reward = 0 (then we finished the game successfully) or check for terminated = True which also means success
        if 0 in rb[:]["next", "reward"]: # Check if we have a zero reward
            zero_reward_index = (rb[:]["next", "reward"] == 0).nonzero(as_tuple=True)[0]
            #print(f"Index of zero reward: {zero_reward_index}")
        terminated_indices = (rb[:]["next", "terminated"] == True).nonzero(as_tuple=True)[0] # Check if we have a terminated episode
        if len(terminated_indices) > 0 and terminated_indices[-1] > last_indice:
            # We have a new terminated episode
            last_indice = terminated_indices[-1]
            print(f"Index of terminated indices: {terminated_indices}")
            # Get the minimum length of the terminated episodes (We still use max_length to keep the same name)
            max_length = rb[:]["next", "step_count"][terminated_indices].min()
            print(f'last length: {rb[:]["next", "step_count"][terminated_indices[-1]]}')
            print(f"Min length: {max_length}")
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            if game_name == "CartPole-v1":
                sample = rb.sample(128).to(device)
            elif game_name == "Acrobot-v1":
                sample = rb.sample(64).to(device)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 100 == 0:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}  i={i}, loss={loss_vals['loss'].item()} exploration={exploration_module.eps}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if game_name == "CartPole-v1" and max_length > 200:
        break
    if game_name == "Acrobot-v1" and max_length < 100 and len(terminated_indices) > 500:
        #print(rb)
        print(f"Solved after {total_count} steps, {total_episodes} episodes with max_length = {max_length}.")
        break
    if game_name == "Acrobot-v1" and i > 100_000:
        print(f"Solved but not learning any more after {total_count} steps, {total_episodes} episodes with max_length = {max_length}.")
        break

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)


# ----- Save the trained model -----
model_save_path = os.path.join("./training_loop", "policy_model.pt")
torch.save(policy.state_dict(), model_save_path)
torchrl_logger.info(f"Model saved to {model_save_path}")


#################################
# Rendering
# ---------
#
# Finally, we run the environment for as many steps as we can and save the
# video locally (notice that we are not exploring).

trajectory = record_env.rollout(max_steps=1000, policy=policy)
video_recorder.dump()
# print the results
print(f"Inference trajectory: {trajectory}")