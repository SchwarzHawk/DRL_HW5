import math
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Gridworld import Gridworld

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ==========================================
# 1. NoisyLinear Layer (取代 epsilon-greedy 的隨機探索)
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ==========================================
# 2. Rainbow Network (Dueling + Distributional + Noisy)
# ==========================================
class RainbowNet(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms=51, Vmin=-15, Vmax=15):
        super(RainbowNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        
        self.linear1 = nn.Linear(num_inputs, 150)
        
        # Dueling 架構搭配 NoisyLinear
        self.adv_hidden = NoisyLinear(150, 100)
        self.adv_out = NoisyLinear(100, num_actions * num_atoms)
        
        self.val_hidden = NoisyLinear(150, 100)
        self.val_out = NoisyLinear(100, num_atoms)
        
        self.register_buffer('supports', torch.linspace(Vmin, Vmax, num_atoms))

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.linear1(x))
        
        adv = F.relu(self.adv_hidden(x))
        val = F.relu(self.val_hidden(x))
        
        adv = self.adv_out(adv).view(batch_size, self.num_actions, self.num_atoms)
        val = self.val_out(val).view(batch_size, 1, self.num_atoms)
        
        # Dueling 計算公式: Q = V + A - mean(A)
        q_dist = val + adv - adv.mean(dim=1, keepdim=True)
        # Softmax 輸出 51 個原子的機率分佈
        prob = F.softmax(q_dist, dim=2)
        return prob

    def act(self, x):
        prob = self.forward(x)
        expected_q = (prob * self.supports).sum(dim=2)
        action = expected_q.argmax(dim=1)
        return action

    def reset_noise(self):
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
        self.val_hidden.reset_noise()
        self.val_out.reset_noise()

# ==========================================
# 3. SumTree & Prioritized Replay Buffer (PER)
# ==========================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, p)
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=50000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.size = 0

    def add(self, experience, error):
        p = (np.abs(error) + 1e-5) ** self.alpha
        self.tree.add(p, experience)
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + 1e-5) ** self.alpha
            self.tree.update(idx, p)

# ==========================================
# 4. Distributional Projection (Categorical DQN - C51)
# ==========================================
def projection_distribution(next_state_prob, rewards, dones, gamma, supports, Vmin, Vmax, num_atoms):
    batch_size = next_state_prob.size(0)
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    tz = rewards + (1 - dones) * gamma * supports.unsqueeze(0)
    tz = tz.clamp(min=Vmin, max=Vmax)
    
    b = (tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    proj_dist = torch.zeros((batch_size, num_atoms), dtype=torch.float32)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
    
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_state_prob * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_state_prob * (b - l.float())).view(-1))
    
    return proj_dist

# ==========================================
# 5. Training Loop
# ==========================================
def train_rainbow(epochs=1500, mem_size=2000, batch_size=64, sync_freq=200):
    num_atoms = 51
    Vmin, Vmax = -15, 15
    n_step = 3
    gamma = 0.99
    
    online_net = RainbowNet(64, 4, num_atoms, Vmin, Vmax)
    target_net = RainbowNet(64, 4, num_atoms, Vmin, Vmax)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=5e-4)
    replay = PrioritizedReplayBuffer(mem_size)
    
    losses = []
    win_count = 0
    steps = 0
    max_moves = 50
    n_step_buffer = deque(maxlen=n_step)
    
    for epoch in range(epochs):
        game = Gridworld(size=4, mode='random')
        state = game.board.render_np().reshape(64) + np.random.rand(64) / 100.0
        status = 1
        mov = 0
        n_step_buffer.clear()
        
        while status == 1:
            mov += 1
            steps += 1
            
            # 由於使用了 NoisyLinear，不再需要 epsilon-greedy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_idx = online_net.act(state_tensor).item()
            action = action_set[action_idx]
            
            game.makeMove(action)
            next_state = game.board.render_np().reshape(64) + np.random.rand(64) / 100.0
            reward = game.reward()
            done = 1.0 if reward > 0 else 0.0
            
            n_step_buffer.append((state, action_idx, reward, next_state, done))
            
            # N-step Experience Replay
            if len(n_step_buffer) == n_step or (done and len(n_step_buffer) > 0):
                R = sum([n_step_buffer[i][2] * (gamma ** i) for i in range(len(n_step_buffer))])
                S, A, _, _, _ = n_step_buffer[0]
                _, _, _, S_n, D_n = n_step_buffer[-1]
                replay.add((S, A, R, S_n, D_n), 10.0) # 初始加入時給予最大的 TD Error 權重
            
            state = next_state
            
            if replay.size > batch_size:
                batch, idxs, is_weights = replay.sample(batch_size)
                state_batch = torch.FloatTensor(np.array([x[0] for x in batch]))
                action_batch = torch.LongTensor(np.array([x[1] for x in batch]))
                reward_batch = torch.FloatTensor(np.array([x[2] for x in batch]))
                next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch]))
                done_batch = torch.FloatTensor(np.array([x[4] for x in batch]))
                is_weights_tensor = torch.FloatTensor(is_weights).unsqueeze(1)
                
                # 重設隨機雜訊
                online_net.reset_noise()
                target_net.reset_noise()
                
                # Double DQN 運算與 C51 分佈投影
                with torch.no_grad():
                    next_actions = online_net.act(next_state_batch)
                    next_dist = target_net(next_state_batch)
                    next_action_dist = next_dist[range(batch_size), next_actions, :]
                    target_dist = projection_distribution(
                        next_action_dist, reward_batch, done_batch, 
                        gamma ** n_step, online_net.supports, Vmin, Vmax, num_atoms
                    )
                
                dist = online_net(state_batch)
                action_dist = dist[range(batch_size), action_batch, :]
                action_dist.data.clamp_(1e-5, 1 - 1e-5) # 避免 log(0) 發生 NaN
                
                # Cross Entropy 作為損失函數與 TD Error
                loss = -(target_dist * action_dist.log()).sum(dim=1)
                replay.update_priorities(idxs, loss.detach().numpy())
                
                # 乘上 Importance Sampling (IS) 權重
                loss = (loss.unsqueeze(1) * is_weights_tensor).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if steps % sync_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())
            
            if reward != -1 or mov > max_moves:
                status = 0
                if reward > 0:
                    win_count += 1
                mov = 0
                
    return online_net, losses

# ==========================================
# 6. Test Agent
# ==========================================
def test_rainbow(model, test_games=1000):
    wins = 0
    max_moves = 15
    model.eval() # 評估時關閉 NoisyNet 的雜訊干擾
    for _ in range(test_games):
        test_game = Gridworld(size=4, mode='random')
        state = test_game.board.render_np().reshape(64) + np.random.rand(64) / 100.0
        status = 1
        mov = 0
        while status == 1:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_idx = model.act(state_tensor).item()
            action = action_set[action_idx]
            test_game.makeMove(action)
            
            state = test_game.board.render_np().reshape(64) + np.random.rand(64) / 100.0
            reward = test_game.reward()
            
            if reward != -1:
                status = 0
                if reward > 0:
                    wins += 1
            mov += 1
            if mov > max_moves:
                status = 0
                
    win_rate = wins / test_games
    return win_rate

def moving_average(a, n=100):
    if len(a) < n:
        return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    print("Training Rainbow DQN (Full Version) in Random Mode...")
    # 開始訓練，Rainbow 收斂效率極高，通常 1000-1500 epochs 即可見效
    model, losses = train_rainbow(epochs=1500)
    
    print("Testing Rainbow DQN...")
    win_rate = test_rainbow(model, test_games=1000)
    print(f"Rainbow DQN Test Win Rate: {win_rate*100:.2f}%")
    
    # 畫出 Loss 曲線
    plt.figure(figsize=(10,6))
    if len(losses) > 0:
        plt.plot(moving_average(losses, n=200), label='Rainbow DQN', color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross Entropy Loss (Moving Average)')
    plt.title('Rainbow DQN Training Loss in Random Mode')
    plt.legend()
    plt.grid(True)
    plt.savefig('rainbow_dqn_loss.png')
    print("Saved training loss to rainbow_dqn_loss.png")
