import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x

class DRRN_Q(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_Q, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        # self.hidden       = nn.Linear(5*hidden_dim, hidden_dim) ######### MODIFIED
        self.hidden       = nn.Linear(3*hidden_dim, hidden_dim)

        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=DEFAULT_DEVICE)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(DEFAULT_DEVICE)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    # def forward(self, taskdes_batch, look_batch, inv_batch, obs_batch, act_batch): ------------------- #MODIFIED
    def forward(self, taskdes_batch, obs_batch, act_batch):
    
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        # look_out = self.packed_rnn(look_batch, self.look_encoder)
        # inv_out = self.packed_rnn(inv_batch, self.inv_encoder)
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        # Expand the state to match the batches of actions
        # z = torch.cat([taskdes_out,look_out, inv_out, obs_out, act_out], dim=1) # Concat along hidden_dim ------------------- #MODIFIED
        z = torch.cat([taskdes_out, obs_out, act_out], dim=1) # Concat along hidden_dim

        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        return act_values



class DRRN_V(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_V, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.hidden       = nn.Linear(4*hidden_dim, hidden_dim) ############### MODIFIED
        self.hidden       = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=DEFAULT_DEVICE)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(DEFAULT_DEVICE)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    # def forward(self, taskdes_batch, look_batch, inv_batch, obs_batch): # MODIFIED
    def forward(self, taskdes_batch, obs_batch):
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        # look_out = self.packed_rnn(look_batch, self.look_encoder) ------------------------- # MODIFIED
        # inv_out = self.packed_rnn(inv_batch, self.inv_encoder)    ------------------------- # MODIFIED
        # z = torch.cat([taskdes_out,look_out, inv_out, obs_out], dim=1) ------------------------- # MODIFIED
        z = torch.cat([taskdes_out, obs_out], dim=1)
        z = F.relu(self.hidden(z))
        act_values = self.scorer(z).squeeze(-1)
        return act_values




def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


class ImplicitQLearning(nn.Module):
    def __init__(self, args, optimizer_factory):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.qf = DRRN_Q(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = DRRN_V(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.tau = args.tau
        self.beta = args.beta
        self.discount = args.discount
        self.alpha = args.alpha

    # def update(self,observations, taskDes, freelook=None, inv=None,  actions, next_observations, next_look, next_inv, rewards, terminals):
    def update(self,observations, taskDes,  actions, next_observations, rewards, terminals):
        obs_ids = [self.sp.EncodeAsIds(o) for o in observations]
        task_ids = [self.sp.EncodeAsIds(t) for t in taskDes]
        # free_ids = [self.sp.EncodeAsIds(f) for f in freelook]
        # inv_ids = [self.sp.EncodeAsIds(i) for i in inv]
        # TextWorld
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        nextobs_ids = [self.sp.EncodeAsIds(next_ob) for next_ob in next_observations]
        # nextfree_ids = [self.sp.EncodeAsIds(f) for f in next_look]
        # nextinv_ids = [self.sp.EncodeAsIds(i) for i in next_inv]
        rewards = torch.tensor(rewards).to(DEFAULT_DEVICE)
        terminals = torch.tensor(terminals).to(DEFAULT_DEVICE)

        with torch.no_grad():
            # target_q = self.q_target(task_ids, free_ids, inv_ids, obs_ids, action_ids) ------ ^^^^^^^^^^
            target_q = self.q_target(task_ids, obs_ids, action_ids)
            # next_v = self.vf(task_ids, nextfree_ids, nextinv_ids, nextobs_ids). --------- ^^^^^^^^^^
            next_v = self.vf(task_ids, nextobs_ids)


        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        # v = self.vf(task_ids, free_ids, inv_ids, obs_ids) ----------------- ^^^^^^^^^^
        v = self.vf(task_ids, obs_ids)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        with open('loss_file.txt', 'a') as f:
            f.write("v_loss is"+str(v_loss))
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # print(rewards.shape)
        # print(terminals.shape)
        # print(next_v.shape)
        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        # qs = self.qf(task_ids, free_ids, inv_ids, obs_ids, action_ids) --------------------- ^^^^^^^^^^^^^^^^^^
        qs = self.qf(task_ids, obs_ids, action_ids)
        # print(qs.shape)
        # print(targets.shape)
        q_loss = F.mse_loss(qs.float(), targets.float())
        # with open('loss_file.txt', 'a') as f:
        #     f.write("q_loss is"+str(q_loss))
        #     f.write('\n')
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

    def act(self, taskDes, freelook, inv, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        free_ids = [self.sp.EncodeAsIds(freelook) for i in range(len(actions))]
        inv_ids = [self.sp.EncodeAsIds(inv) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        q_value = list(self.q_target(task_ids, free_ids, inv_ids, obs_ids, action_ids))
        lst = [i for i in zip(q_value, actions)]
        lst.sort(reverse=True)
        return lst[0][1]
    
    # def get_q(self, taskDes, freelook, inv, observation, actions):
    def get_q(self, taskDes, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        # free_ids = [self.sp.EncodeAsIds(freelook) for i in range(len(actions))]
        # inv_ids = [self.sp.EncodeAsIds(inv) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        q_value = self.q_target(task_ids, obs_ids, action_ids)
        v_value = self.vf(task_ids, obs_ids)
        return q_value-v_value