import torch
from tqdm import trange
import numpy as np
import pandas as pd
import random
from alfworld_iql import ImplicitQLearning


#utils.py
def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}

def sample_batch_all(dataset, start, end):
    k = list(dataset.keys())[0]
    n = len(dataset[k])
    for v in dataset.values():
        assert len(v) == n
    return {k: v[start:end] for k, v in dataset.items()}


def reform_datapd(data_pd):
    lst = list(data_pd['terminal'].values)
    end_lst = [-1]
    for i in range(3498):
        if lst[i] != 0:
            end_lst.append(i)
    for end in range(1,len(end_lst)):
        # print(f'end_lst[end] = {end_lst[end]}')
        reward = data_pd.at[end_lst[end],'reward']
        length = end_lst[end]-end_lst[end-1]
        # print(f'reward = {reward}')
        # print(f'type(reward) = {type(reward)}')
        # print(f'type(length) = {type(length)}')
        avg_reward = reward/length
        for j in range(end_lst[end-1]+1,end_lst[end]+1):
            data_pd.at[j,'reward'] = avg_reward
    return

def get_dataset_alfworld():
    dataset = {"observations":[], 'taskDes':[], "actions":[], "next_observations":[], "rewards":[], "terminals":[]}
    # data_pd.columns = ['taskDes','observation','next_observation','reward','terminal','action']
    columns = ['taskDes','observation','next_observation','reward','terminal','action']
    data_pd = pd.read_csv("train_alf_trajectories.csv",header=None, names=columns, skiprows=1)
    reform_datapd(data_pd)
    data_pd = data_pd.sample(frac=1)
    dataset['taskDes'] = list(data_pd['taskDes'].values)
    dataset["observations"] = list(data_pd['observation'].values)
    dataset["actions"] = list(data_pd['action'].values)
    dataset["next_observations"] = list(data_pd["next_observation"].values)
    dataset["rewards"] = list(data_pd['reward'].values)
    dataset["terminals"] = list(data_pd['terminal'].values)
    return dataset





def train_iql_alf(args):
    torch.set_num_threads(1)
    set_seed(args.seed, env=None)
    dataset = get_dataset_alfworld()
    iql = ImplicitQLearning(
        args,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate)
    )
    start = 0
    k = list(dataset.keys())[0]
    n = len(dataset[k])
    print(n)
    round = 0
    for step in trange(args.n_steps):
        if round >= 2:
            break
        end = start + args.batch_size
        if end >= n:
            iql.update(**sample_batch_all(dataset, start, n))
            start = 0
            round += 1
            print('End round%d\n'%round)
        else:
            # print(sample_batch_all(dataset, start, end))
            iql.update(**sample_batch_all(dataset, start, end))
            start = end
        print("Start = %d"%start)

    model_path = 'final_iql_alfworld.pt'
    torch.save(iql.state_dict(), model_path)





args = {
    "output_dir": "logs",
    "spm_path": "unigram_8k.model",
    "seed": 0,
    "discount": 0.99,
    "hidden_dim": 128,
    "embedding_dim": 64,
    "n_hidden": 2,
    "n_steps": 20000,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "alpha": 0.005,
    "tau": 0.7,
    "beta": 3.0,
    "deterministic_policy": False,  # Boolean flag
    "eval_period": 10000,
    "n_eval_episodes": 10,
    "max_episode_steps": 1000,
    "env_name": "alfworld",
    "dataset_type": "gpt"}


from types import SimpleNamespace
args = SimpleNamespace(**args)


train_iql_alf(args)