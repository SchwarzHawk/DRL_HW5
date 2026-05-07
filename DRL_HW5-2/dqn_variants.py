import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Gridworld import Gridworld

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

# -------------------------------
# 1. Basic Q-Network
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# 2. Dueling Q-Network
# -------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        
        # 將網路拆分出 Value 與 Advantage
        self.value_stream = nn.Linear(100, 1)
        self.advantage_stream = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# -------------------------------
# Training Logic
# -------------------------------
def train_agent(model_type='basic', epochs=1000, mem_size=1000, batch_size=200, sync_freq=500):
    """
    model_type: 'basic', 'double', 'dueling'
    """
    if model_type == 'dueling':
        online_net = DuelingQNetwork()
        target_net = DuelingQNetwork()
    else:
        online_net = QNetwork()
        target_net = QNetwork()
        
    target_net.load_state_dict(online_net.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)
    
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    
    replay = deque(maxlen=mem_size)
    max_moves = 50
    
    losses = []
    win_count = 0
    
    steps = 0
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while status == 1:
            mov += 1
            steps += 1
            
            # Epsilon-greedy action selection
            qval = online_net(state1)
            qval_ = qval.data.numpy()
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = online_net(state1_batch)
                
                with torch.no_grad():
                    if model_type == 'double':
                        # Double DQN: Online network chooses action, Target network evaluates it
                        online_Q2 = online_net(state2_batch)
                        best_actions = torch.argmax(online_Q2, dim=1).unsqueeze(dim=1)
                        target_Q2 = target_net(state2_batch)
                        maxQ = target_Q2.gather(dim=1, index=best_actions).squeeze()
                    else:
                        # Basic DQN & Dueling DQN: Target network chooses and evaluates action
                        target_Q2 = target_net(state2_batch)
                        maxQ = torch.max(target_Q2, dim=1)[0]
                
                Y = reward_batch + gamma * ((1 - done_batch) * maxQ)
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if steps % sync_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())
            
            if reward != -1 or mov > max_moves:
                status = 0
                if reward > 0:
                    win_count += 1
                mov = 0
                
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
    return online_net, losses, win_count

# -------------------------------
# Test Agent Function
# -------------------------------
def test_agent(model, test_games=1000):
    wins = 0
    max_moves = 15
    for _ in range(test_games):
        test_game = Gridworld(size=4, mode='player')
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state = torch.from_numpy(state_).float()
        status = 1
        mov = 0
        while status == 1:
            qval = model(state)
            action_ = np.argmax(qval.data.numpy())
            action = action_set[action_]
            test_game.makeMove(action)
            
            state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state = torch.from_numpy(state_).float()
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
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    epochs = 2000 # 為了有足夠的資料來收斂，設定回合數
    print("Training Basic DQN...")
    basic_model, basic_losses, basic_train_wins = train_agent('basic', epochs=epochs)
    basic_win_rate = test_agent(basic_model)
    print(f"Basic DQN Test Win Rate: {basic_win_rate*100:.2f}%")
    
    print("Training Double DQN...")
    double_model, double_losses, double_train_wins = train_agent('double', epochs=epochs)
    double_win_rate = test_agent(double_model)
    print(f"Double DQN Test Win Rate: {double_win_rate*100:.2f}%")
    
    print("Training Dueling DQN...")
    dueling_model, dueling_losses, dueling_train_wins = train_agent('dueling', epochs=epochs)
    dueling_win_rate = test_agent(dueling_model)
    print(f"Dueling DQN Test Win Rate: {dueling_win_rate*100:.2f}%")

    # Plot losses
    plt.figure(figsize=(10,6))
    if len(basic_losses) > 0:
        plt.plot(moving_average(basic_losses), label='Basic DQN', alpha=0.7)
    if len(double_losses) > 0:
        plt.plot(moving_average(double_losses), label='Double DQN', alpha=0.7)
    if len(dueling_losses) > 0:
        plt.plot(moving_average(dueling_losses), label='Dueling DQN', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss (Moving Average)')
    plt.title('DQN Variants Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_variants_loss.png')
    print("Saved training loss comparison to dqn_variants_loss.png")
