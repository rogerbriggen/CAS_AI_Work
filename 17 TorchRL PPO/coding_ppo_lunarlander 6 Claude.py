"""
Reinforcement Learning (PPO) with TorchRL for Robust LunarLander
==================================================
This script trains a PPO agent to land the lunar lander across various wind conditions.
"""

import os
import sys
import warnings
import numpy as np
from collections import defaultdict
import random

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_cells = 512  # Larger network for a more complex problem
lr = 1e-4
max_grad_norm = 1.0

# Training parameters
frames_per_batch = 2000  # Reduced to avoid memory issues
total_frames = 1_000_000  # Reduced total training time
sub_batch_size = 128
num_epochs = 10
clip_epsilon = 0.1
gamma = 0.99
lmbda = 0.95
entropy_eps = 5e-3  # Increased entropy for better exploration

# Environment parameters
game_name = "LunarLanderContinuous-v2"

# Wind parameter ranges
WIND_POWER_MIN = 0.0
WIND_POWER_MAX = 25.0
TURBULENCE_POWER_MIN = 0.0
TURBULENCE_POWER_MAX = 3.0

# Number of different environments to use during training
NUM_ENV_VARIATIONS = 4  # Reduced from 8 to avoid memory issues

class WindParameterizer:
    """Handles wind parameter randomization for creating environment variations"""
    
    def __init__(self, 
                 wind_power_range=(WIND_POWER_MIN, WIND_POWER_MAX),
                 turbulence_range=(TURBULENCE_POWER_MIN, TURBULENCE_POWER_MAX)):
        self.wind_power_range = wind_power_range
        self.turbulence_range = turbulence_range
        
        # For fixed parameter sampling
        self.fixed_variations = self._create_fixed_variations(NUM_ENV_VARIATIONS)
    
    def _create_fixed_variations(self, num_variations):
        """Create a set of fixed parameter variations to use during training"""
        variations = []
        
        # Generate evenly spaced parameters across the ranges
        for i in range(num_variations):
            if i == 0:
                # First variation has no wind (for learning fundamentals)
                wind_power = 0.0
                turbulence = 0.0
            else:
                # Distribute remaining variations across parameter space
                wind_power = self.wind_power_range[0] + (i / (num_variations - 1)) * (
                    self.wind_power_range[1] - self.wind_power_range[0])
                turbulence = self.turbulence_range[0] + (i / (num_variations - 1)) * (
                    self.turbulence_range[1] - self.turbulence_range[0])
            
            variations.append({
                'enable_wind': True if wind_power > 0 else False,
                'wind_power': wind_power,
                'turbulence_power': turbulence
            })
        
        return variations
        
    def get_fixed_variation(self, idx):
        """Get a specific variation by index"""
        return self.fixed_variations[idx % len(self.fixed_variations)]
    
    def sample_random_parameters(self):
        """Sample random wind parameters within the specified ranges"""
        enable_wind = True if random.random() > 0.1 else False
        wind_power = random.uniform(*self.wind_power_range) if enable_wind else 0.0
        turbulence = random.uniform(*self.turbulence_range) if enable_wind else 0.0
        
        return {
            'enable_wind': enable_wind,
            'wind_power': wind_power,
            'turbulence_power': turbulence
        }


def create_env(wind_params=None, transform=True, record=False):
    """Create a LunarLander environment with specified wind parameters"""
    extra = {}
    
    # Set wind parameters if provided
    if wind_params:
        extra.update(wind_params)
    
    # Create base environment
    base_env = GymEnv(game_name, device=device, **extra)
    
    if record:
        # Create recording environment for evaluation
        logger = CSVLogger(exp_name="ppo_lunar_robust", log_dir="./training_loop", video_format="mp4")
        video_recorder = VideoRecorder(logger, tag=f"video_wind_{wind_params.get('wind_power', 0)}")
        
        env = TransformedEnv(
            GymEnv(game_name, from_pixels=True, pixels_only=False, device=device, **extra),
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
                video_recorder
            )
        ).to(device)
        
        return env, video_recorder
    
    if transform:
        # Create transformed environment for training
        env = TransformedEnv(
            base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        ).to(device)
        
        return env
    
    return base_env


class MultiEnvCollector:
    """A wrapper to collect data from multiple environment variations"""
    
    def __init__(self, parameterizer, policy_module, frames_per_batch, total_frames, device):
        self.parameterizer = parameterizer
        self.policy_module = policy_module
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.device = device
        self.env_collectors = []
        self.frames_collected = 0
        
        # Create collectors for each fixed variation
        for i in range(NUM_ENV_VARIATIONS):
            wind_params = parameterizer.get_fixed_variation(i)
            env = create_env(wind_params)
            
            # Initialize normalization stats
            env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
            
            collector = SyncDataCollector(
                env,
                policy_module,
                frames_per_batch=frames_per_batch // NUM_ENV_VARIATIONS,  # Split batch across envs
                total_frames=total_frames,
                split_trajs=False,
                device=device,
            )
            
            self.env_collectors.append((collector, wind_params))
    
    def collect_batch(self):
        """Collect a batch of data from different environment variations"""
        combined_tensordict = None
        
        # Collect data from each environment variation
        for i, (collector, wind_params) in enumerate(self.env_collectors):
            try:
                # Get next batch from this collector
                tensordict_data = next(iter(collector))
                
                # Add wind parameters as metadata
                tensordict_data.set("wind_power", torch.tensor([wind_params['wind_power']], device=device).expand(tensordict_data.batch_size))
                tensordict_data.set("turbulence_power", torch.tensor([wind_params['turbulence_power']], device=device).expand(tensordict_data.batch_size))
                
                if combined_tensordict is None:
                    combined_tensordict = tensordict_data
                else:
                    # Concatenate with data from other environments
                    combined_tensordict = torch.cat([combined_tensordict, tensordict_data], dim=0)
                
            except StopIteration:
                # This collector is exhausted, create a new one with random wind parameters
                wind_params = self.parameterizer.sample_random_parameters()
                env = create_env(wind_params)
                env.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
                
                new_collector = SyncDataCollector(
                    env,
                    self.policy_module,
                    frames_per_batch=frames_per_batch // NUM_ENV_VARIATIONS,
                    total_frames=total_frames,
                    split_trajs=False,
                    device=device,
                )
                
                # Replace the exhausted collector
                self.env_collectors[i] = (new_collector, wind_params)
            except Exception as e:
                print(f"Error collecting data from environment {i} with wind {wind_params}: {e}")
                # Create a new collector for this environment variation
                new_wind_params = self.parameterizer.sample_random_parameters()
                env = create_env(new_wind_params)
                env.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
                
                new_collector = SyncDataCollector(
                    env,
                    self.policy_module,
                    frames_per_batch=frames_per_batch // NUM_ENV_VARIATIONS,
                    total_frames=total_frames,
                    split_trajs=False,
                    device=device,
                )
                
                # Replace the problematic collector
                self.env_collectors[i] = (new_collector, new_wind_params)
        
        if combined_tensordict is None:
            # If all collectors failed, create a new emergency collector with default settings
            print("All collectors failed. Creating emergency collector.")
            wind_params = {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0}
            env = create_env(wind_params)
            env.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
            
            emergency_collector = SyncDataCollector(
                env,
                self.policy_module,
                frames_per_batch=frames_per_batch,
                total_frames=total_frames,
                split_trajs=False,
                device=device,
            )
            
            tensordict_data = next(iter(emergency_collector))
            combined_tensordict = tensordict_data
            self.env_collectors[0] = (emergency_collector, wind_params)
        
        self.frames_collected += combined_tensordict.numel()
        return combined_tensordict
    
    def is_complete(self):
        """Check if we've collected the total number of frames"""
        return self.frames_collected >= self.total_frames


def create_model():
    """Create policy and value networks"""
    # For reference, we need the action spec from an environment
    base_env = create_env(transform=False)
    
    # Create actor network (policy) - Using fixed sized layers instead of lazy
    input_dim = base_env.observation_spec["observation"].shape[-1]
    output_dim = base_env.action_spec.shape[-1]
    
    actor_net = nn.Sequential(
        nn.Linear(input_dim, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 128, device=device),
        nn.ReLU(),
        nn.Linear(128, 2 * output_dim, device=device),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=base_env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": base_env.action_spec_unbatched.space.low,
            "high": base_env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )
    
    # Create value network (critic) - Using fixed sized layers
    value_net = nn.Sequential(
        nn.Linear(input_dim, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 128, device=device),
        nn.ReLU(),
        nn.Linear(128, 1, device=device),
    )
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    
    # With fixed-size networks, we don't need the dummy initialization anymore
    # but we'll keep a simpler version just to make sure everything's working
    dummy_env = create_env(transform=True)
    dummy_env.transform[0].init_stats(num_iter=10, reduce_dim=0, cat_dim=0)
    
    return policy_module, value_module, base_env.action_spec


def evaluate_policy(policy_module, wind_parameterizer, num_eval_runs=5):
    """Evaluate policy across multiple wind conditions"""
    eval_results = []
    
    # Make sure policy is in eval mode
    policy_module.eval()
    
    # Test on various wind conditions
    test_conditions = [
        {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0},  # No wind
        {'enable_wind': True, 'wind_power': 5.0, 'turbulence_power': 0.5},   # Light wind
        {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.5},  # Medium wind
        {'enable_wind': True, 'wind_power': 25.0, 'turbulence_power': 2.5},  # Strong wind
    ]
    
    for condition in test_conditions:
        condition_rewards = []
        
        # Create and prepare environment for this condition
        env = create_env(condition)
        env.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
        
        # Run multiple evaluations for each condition
        for _ in range(num_eval_runs):
            try:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(1000, policy_module)
                    condition_rewards.append(eval_rollout["next", "reward"].sum().item())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                # In case of error, add a very negative reward
                condition_rewards.append(-1000)
        
        # Store average reward for this condition
        eval_results.append({
            'wind_power': condition['wind_power'],
            'turbulence_power': condition['turbulence_power'],
            'avg_reward': sum(condition_rewards) / len(condition_rewards) if condition_rewards else -1000,
            'min_reward': min(condition_rewards) if condition_rewards else -1000,
            'max_reward': max(condition_rewards) if condition_rewards else -1000
        })
    
    # Set policy back to training mode
    policy_module.train()
    return eval_results


def train():
    """Main training function"""
    # Create wind parameterizer
    wind_parameterizer = WindParameterizer()
    
    # Create policy and value networks
    policy_module, value_module, action_spec = create_model()
    
    # Create multi-environment collector
    multi_collector = MultiEnvCollector(
        wind_parameterizer,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # Create loss modules directly without dummy data (we're using fixed-size networks now)
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )
    
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    # Create optimizer and scheduler
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )
    
    # Setup logging
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    best_reward = float('-inf')
    best_model_path = None
    
    # Create TensorBoard writer
    writer = SummaryWriter(comment="-ppo_lunar_robust_wind")
    
    # Log hyperparameters and seeds
    writer.add_text('Random Seeds', f'PyTorch Seed: {SEED}')
    
    hyperparams = {
        "num_cells": num_cells,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "frames_per_batch": frames_per_batch,
        "total_frames": total_frames,
        "sub_batch_size": sub_batch_size,
        "num_epochs": num_epochs,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "lmbda": lmbda,
        "entropy_eps": entropy_eps,
        "wind_power_range": f"{WIND_POWER_MIN}-{WIND_POWER_MAX}",
        "turbulence_range": f"{TURBULENCE_POWER_MIN}-{TURBULENCE_POWER_MAX}",
        "num_env_variations": NUM_ENV_VARIATIONS
    }
    
    for key, value in hyperparams.items():
        writer.add_text(f"Hyperparameters/{key}", str(value))
    
    # Training loop
    step_idx = 0
    num_batches = 0
    
    try:
        while not multi_collector.is_complete():
            # Collect batch of data from multiple environments
            tensordict_data = multi_collector.collect_batch()
            num_batches += 1
        
        # Train on this batch for multiple epochs
        for _ in range(num_epochs):
            # Compute advantage for PPO
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            # Sample and optimize
            for _ in range(frames_per_batch // sub_batch_size):
                step_idx += 1
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                
                # Compute total loss
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                # Log losses
                writer.add_scalar("loss/objective", loss_vals["loss_objective"], step_idx)
                writer.add_scalar("loss/critic", loss_vals["loss_critic"], step_idx)
                writer.add_scalar("loss/entropy", loss_vals["loss_entropy"], step_idx)
                writer.add_scalar("loss/total", loss_value, step_idx)
                
                # Optimization step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        # Log training metrics
        avg_reward = tensordict_data["next", "reward"].mean().item()
        max_steps = tensordict_data["step_count"].max().item()
        
        writer.add_scalar("train/reward", avg_reward, step_idx)
        writer.add_scalar("train/step_count", max_steps, step_idx)
        logs["reward"].append(avg_reward)
        logs["step_count"].append(max_steps)
        
        # Update progress bar
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={avg_reward:.4f} (init={logs['reward'][0]:.4f})"
        stepcount_str = f"step count (max): {max_steps}"
        lr_str = f"lr policy: {optim.param_groups[0]['lr']:.6f}"
        
        # Evaluate periodically
        if num_batches % 10 == 0:
            # Evaluate across different wind conditions
            eval_results = evaluate_policy(policy_module, wind_parameterizer)
            
            # Log evaluation results
            for i, result in enumerate(eval_results):
                condition_name = f"wind_{result['wind_power']}_turb_{result['turbulence_power']}"
                writer.add_scalar(f"eval/reward_{condition_name}", result['avg_reward'], step_idx)
                writer.add_scalar(f"eval/min_reward_{condition_name}", result['min_reward'], step_idx)
                writer.add_scalar(f"eval/max_reward_{condition_name}", result['max_reward'], step_idx)
            
            # Calculate average performance across all conditions
            avg_eval_reward = sum(r['avg_reward'] for r in eval_results) / len(eval_results)
            writer.add_scalar("eval/avg_reward_all_conditions", avg_eval_reward, step_idx)
            logs["eval_reward"].append(avg_eval_reward)
            
            # Save best model based on average performance
            if avg_eval_reward > best_reward:
                best_reward = avg_eval_reward
                best_model_path = os.path.join("./training_loop", f"ppo_lunar_robust_wind_best_{best_reward:.2f}.pt")
                torch.save(policy_module.state_dict(), best_model_path)
                torchrl_logger.info(f"New best model with avg reward {best_reward:.2f} saved to {best_model_path}")
            
            eval_str = f"eval avg reward: {avg_eval_reward:.4f} (best: {best_reward:.4f})"
        
        pbar.set_description(", ".join(filter(None, [eval_str, cum_reward_str, stepcount_str, lr_str])))
        
        # Update learning rate
        scheduler.step()
    
    except Exception as e:
        print(f"Exception during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Close logger and progress bar
    writer.close()
    pbar.close()
    
    # Save final model
    final_model_path = os.path.join("./training_loop", "ppo_lunar_robust_wind_final.pt")
    torch.save(policy_module.state_dict(), final_model_path)
    torchrl_logger.info(f"Final model saved to {final_model_path}")
    
    # Return paths to best and final models
    return best_model_path, final_model_path


def visualize_policy(model_path, wind_parameterizer):
    """Create videos of the trained policy under different wind conditions"""
    # Load policy
    policy_module, _, _ = create_model()
    policy_module.load_state_dict(torch.load(model_path))
    policy_module.eval()
    
    # Test conditions to visualize
    test_conditions = [
        {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0},  # No wind
        {'enable_wind': True, 'wind_power': 10.0, 'turbulence_power': 1.0},  # Medium wind
        {'enable_wind': True, 'wind_power': 25.0, 'turbulence_power': 2.5},  # Strong wind
    ]
    
    video_recorders = []
    
    # Create and record rollouts for each condition
    for condition in test_conditions:
        env, recorder = create_env(condition, transform=True, record=True)
        env.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
        
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            trajectory = env.rollout(max_steps=1000, policy=policy_module)
            print(f"Wind: {condition['wind_power']}, Turbulence: {condition['turbulence_power']}")
            print(f"Total reward: {trajectory['next', 'reward'].sum().item()}")
        
        recorder.dump()
        video_recorders.append(recorder)
    
    return video_recorders


def plot_training_curves(logs):
    """Plot training and evaluation curves"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("Training Rewards (Average)")
    plt.xlabel("Batch")
    plt.ylabel("Reward")
    
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max Step Count (Training)")
    plt.xlabel("Batch")
    plt.ylabel("Steps")
    
    if "eval_reward" in logs and logs["eval_reward"]:
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval_reward"])
        plt.title("Evaluation Rewards (Average Across Wind Conditions)")
        plt.xlabel("Evaluation")
        plt.ylabel("Reward")
    
    plt.tight_layout()
    plt.savefig("./training_loop/training_curves.png")
    plt.show()


if __name__ == "__main__":
    # Create output directory
    os.makedirs("./training_loop", exist_ok=True)
    
    # Train the agent
    best_model_path, final_model_path = train()
    
    # Create wind parameterizer for visualization
    wind_parameterizer = WindParameterizer()
    
    # Visualize the best model
    visualize_policy(best_model_path, wind_parameterizer)