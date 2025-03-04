#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torchrl.envs import EnvBase
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from typing import Optional
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot
from torchrl.data import DiscreteTensorSpec

from tensordict.nn import TensorDictModule, TensorDictSequential


# Specify device explicitly
device = torch.device("cpu")  # or "cuda" if you have GPU support

# Generate Realistic Synthetic Data
#Platzierung:
#Organisch: Erscheint aufgrund des Suchalgorithmus, ohne Bezahlung.
#Paid: Wird aufgrund einer Werbekampagne oder bezahlten Platzierung angezeigt.
#Kosten:
#Organisch: Es fallen in der Regel keine direkten Kosten pro Klick oder Impression an.
#Paid: Werbetreibende zahlen oft pro Klick (CPC) oder pro Impression (CPM = pro Sichtkontakt, unabhängig ob jemand klickt oder nicht).
def generate_synthetic_data(num_samples=1000):
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],  # Eindeutiger Name oder Identifier für das Keyword
        "competitiveness": np.random.uniform(0, 1, num_samples),     # Wettbewerbsfähigkeit des Keywords (Wert zwischen 0 und 1). Je mehr Leute das Keyword wollen, desto näher bei 1 und somit desto teurer.
        "difficulty_score": np.random.uniform(0, 1, num_samples),      # Schwierigkeitsgrad des Keywords organisch gute Platzierung zu erreichen (Wert zwischen 0 und 1). 1 = mehr Aufwand und Optimierung nötig.
        "organic_rank": np.random.randint(1, 11, num_samples),         # Organischer Rang, z.B. Position in Suchergebnissen (1 bis 10)
        "organic_clicks": np.random.randint(50, 5000, num_samples),    # Anzahl der Klicks auf organische Suchergebnisse
        "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),      # Klickrate (CTR) für organische Suchergebnisse
        "paid_clicks": np.random.randint(10, 3000, num_samples),       # Anzahl der Klicks auf bezahlte Anzeigen
        "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),        # Klickrate (CTR) für bezahlte Anzeigen
        "ad_spend": np.random.uniform(10, 10000, num_samples),         # Werbebudget bzw. Ausgaben für Anzeigen
        "ad_conversions": np.random.randint(0, 500, num_samples),      # Anzahl der Conversions (Erfolge) von Anzeigen
        "ad_roas": np.random.uniform(0.5, 5, num_samples),             # Return on Ad Spend (ROAS) für Anzeigen, wobei Werte < 1 Verlust anzeigen
        "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),    # Conversion-Rate (Prozentsatz der Besucher, die konvertieren)
        "cost_per_click": np.random.uniform(0.1, 10, num_samples),     # Kosten pro Klick (CPC)
        "cost_per_acquisition": np.random.uniform(5, 500, num_samples),  # Kosten pro Akquisition (CPA)
        "previous_recommendation": np.random.choice([0, 1], size=num_samples),  # Frühere Empfehlung (0 = nein, 1 = ja)
        "impression_share": np.random.uniform(0.1, 1.0, num_samples),  # Anteil an Impressionen (Sichtbarkeit der Anzeige) im Vergleich mit allen anderen die dieses Keyword wollen
        "conversion_value": np.random.uniform(0, 10000, num_samples)   # Monetärer Wert der Conversions (Ein monetärer Wert, der den finanziellen Nutzen aus den erzielten Conversions widerspiegelt. Dieser Wert gibt an, wie viel Umsatz oder Gewinn durch die Conversions generiert wurde – je höher der Wert, desto wertvoller sind die Conversions aus Marketingsicht.)
    }
    return pd.DataFrame(data)


def getKeywords():
    return ["investments", "stocks", "crypto", "cryptocurrency", "bitcoin", "real estate", "gold", "bonds", "broker", "finance", "trading", "forex", "etf", "investment fund", "investment strategy", "investment advice", "investment portfolio", "investment opportunities", "investment options", "investment calculator", "investment plan", "investment account", "investment return", "investment risk", "investment income", "investment growth", "investment loss", "investment profit", "investment return calculator", "investment return formula", "investment return rate"]


def generateData():
    seed = 42  # or any integer of your choice
    random.seed(seed)      # Sets the seed for the Python random module
    np.random.seed(seed)   # Sets the seed for NumPy's random generator
    torch.manual_seed(seed)  # Sets the seed for PyTorch

    # If you're using CUDA as well, you may also set:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Generate synthetic data
    # Do it 1000 times
    dataset = pd.DataFrame()
    for i in range(1000):
        # append to dataset
        dataset = generate_synthetic_data(len(getKeywords()))
        


# Load synthetic dataset
dataset = generate_synthetic_data(1000)
feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate", "cost_per_click"]


def read_and_organize_csv(file_path):
    df = pd.read_csv(file_path)
    organized_data = pd.DataFrame()

    # Skip the 'step' column
    df = df.drop(columns=['step'])

    # Get unique keywords
    keywords = df['keyword'].unique()

    # Organize data
    for i in range(5000):
        for keyword in keywords:
            keyword_data = df[df['keyword'] == keyword]
            if len(keyword_data) > i:
                organized_data = pd.concat([organized_data, keyword_data.iloc[[i]]])

    return organized_data.reset_index(drop=True)

# Example usage
#organized_dataset = read_and_organize_csv('18 TorchRL Ads/balanced_ad_dataset_real_keywords.csv')
#organized_dataset.to_csv('organized_dataset.csv', index=False)


df = pd.read_csv('18 TorchRL Ads/organized_dataset.csv')
df.head()

def get_entry_from_dataset(df, index):
    # Count unique keywords
    seen_keywords = set()
    if not hasattr(get_entry_from_dataset, "unique_keywords"):
        seen_keywords = set()
        for i, row in df.iterrows():
            keyword = row['keyword']
            if keyword in seen_keywords:
                break
            seen_keywords.add(keyword)
        get_entry_from_dataset.unique_keywords = seen_keywords
        get_entry_from_dataset.keywords_amount = len(seen_keywords)
    else:
        seen_keywords = get_entry_from_dataset.unique_keywords

    keywords_amount = get_entry_from_dataset.keywords_amount
    return df.iloc[index * keywords_amount:index * keywords_amount + keywords_amount].reset_index(drop=True)

# Example usage
entry = get_entry_from_dataset(df, 0)
print(entry)

entry = get_entry_from_dataset(df, 1)
print(entry)


# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset, device="cpu"):
        super().__init__(device=device)
        self.dataset = dataset
        self.num_features = len(feature_columns)
        self.action_spec = OneHot(n=2, dtype=torch.int64)
        self.reset()

    def _reset(self, tensordict=None):
        sample = self.dataset.sample(1)
        state = torch.tensor(sample[feature_columns].values, dtype=torch.float32).squeeze()
        #return TensorDict({"observation": state}, batch_size=[])
        # step_count initialisieren
        return TensorDict(
            {
                "observation": state,
                "step_count": torch.tensor(0, dtype=torch.int64, device=self.device)
            },
            batch_size=[]
        )

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        #action = tensordict["action"].item()
        next_sample = self.dataset.sample(1)
        next_state = torch.tensor(next_sample[feature_columns].values, dtype=torch.float32).squeeze()
        reward = self._compute_reward(action, next_sample)
        done = False
        #return TensorDict({"observation": next_state, "reward": torch.tensor(reward), "done": torch.tensor(done)}, batch_size=[])
         # hole aktuellen step_count (falls vorhanden) und inkrementiere ihn
        current_step = tensordict.get("step_count", torch.tensor(0, dtype=torch.int64, device=self.device))
        new_step = current_step + 1
        return TensorDict(
            {
                "observation": next_state,
                "reward": torch.tensor(reward, device=self.device),
                "done": torch.tensor(done, device=self.device),
                "step_count": new_step,
            },
            batch_size=[]
        )

    def _compute_reward(self, action, sample):
        cost = sample["ad_spend"].values[0]
        ctr = sample["paid_ctr"].values[0]
        if action == 1 and cost > 5000:
            reward = 1.0
        elif action == 0 and ctr > 0.15:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

# Initialize Environment
env = AdOptimizationEnv(dataset, device=device)
state_dim = env.num_features
action_dim = env.action_spec.n




# In[ ]:


env.action_spec


# In[ ]:


from torchrl.modules import EGreedyModule, MLP, QValueModule

value_mlp = MLP(in_features=env.num_features, out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))
# Make sure your policy is on the correct device
policy = policy.to(device)

exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
exploration_module = exploration_module.to(device)
policy_explore = TensorDictSequential(policy, exploration_module)


# In[ ]:


value_mlp


# In[ ]:


from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

from torch.optim import Adam


# In[ ]:


from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)


# In[ ]:


import time
total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data.to(device))
    #max_length = rb[:]["next", "step_count"].max()
    max_length = rb["step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            # Make sure sample is on the correct device
            sample = sample.to(device)  # Move the sample to the specified device
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10 == 0: # Fixed condition (was missing '== 0')
                print(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

print(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)


# In[ ]:




