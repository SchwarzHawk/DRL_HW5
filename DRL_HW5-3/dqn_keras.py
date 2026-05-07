import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque
import matplotlib.pyplot as plt

# 加入上一層目錄以利匯入 Gridworld
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
# 1. Build Keras Q-Network
# -------------------------------
def build_q_network():
    model = models.Sequential([
        layers.Dense(150, activation='relu', input_shape=(64,)),
        layers.Dense(100, activation='relu'),
        layers.Dense(4, activation='linear')
    ])
    return model

# -------------------------------
# Training Logic (Basic DQN with enhancements)
# -------------------------------
def train_agent_keras(epochs=1500, mem_size=2000, batch_size=200, sync_freq=500):
    online_net = build_q_network()
    target_net = build_q_network()
    target_net.set_weights(online_net.get_weights())
    
    # 訓練技巧 1: Learning Rate Scheduling (學習率排程)
    # 初始學習率較大，並隨著步數呈指數遞減，確保訓練後期更為穩定
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=True
    )
    
    # 訓練技巧 2: Gradient Clipping (梯度裁剪)
    # 設定 clipnorm=1.0 來避免損失函數劇烈變化導致的梯度爆炸問題
    optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    gamma = 0.9
    
    # 訓練技巧 3: Epsilon Decay (探索機率衰減)
    # 讓平滑的衰減涵蓋更多回合，確保模型有足夠時間探索複雜的 random 模式
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998 # 衰減率放緩
    
    replay = deque(maxlen=mem_size)
    max_moves = 50
    
    losses = []
    win_count = 0
    steps = 0
    
    for i in range(epochs):
        # 使用難度最高的 random 模式
        game = Gridworld(size=4, mode='random')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        status = 1
        mov = 0
        
        while status == 1:
            mov += 1
            steps += 1
            
            # Epsilon-greedy 動作選擇
            state1_tf = tf.convert_to_tensor(state1_, dtype=tf.float32)
            qval = online_net(state1_tf)
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval.numpy()[0])
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1_[0], action_, reward, state2_[0], done)
            replay.append(exp)
            state1_ = state2_
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = np.array([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = np.array([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = np.array([r for (s1, a, r, s2, d) in minibatch], dtype=np.float32)
                state2_batch = np.array([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = np.array([d for (s1, a, r, s2, d) in minibatch], dtype=np.float32)
                
                # 將 numpy array 轉換為 tensor
                s1_tensor = tf.convert_to_tensor(state1_batch, dtype=tf.float32)
                s2_tensor = tf.convert_to_tensor(state2_batch, dtype=tf.float32)
                
                # Target 計算
                target_Q2 = target_net(s2_tensor)
                maxQ = tf.reduce_max(target_Q2, axis=1)
                Y = reward_batch + gamma * ((1 - done_batch) * maxQ)
                
                # 計算 Loss 並進行反向傳播 (Gradient Tape)
                with tf.GradientTape() as tape:
                    Q1 = online_net(s1_tensor)
                    # 萃取出被選中動作的 Q 值
                    action_masks = tf.one_hot(action_batch, 4)
                    X = tf.reduce_sum(Q1 * action_masks, axis=1)
                    
                    loss = loss_fn(Y, X)
                
                # 更新權重
                grads = tape.gradient(loss, online_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, online_net.trainable_variables))
                losses.append(loss.numpy())
                
                # 更新 Target Network
                if steps % sync_freq == 0:
                    target_net.set_weights(online_net.get_weights())
            
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
        test_game = Gridworld(size=4, mode='random')
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        status = 1
        mov = 0
        while status == 1:
            state_tf = tf.convert_to_tensor(state_, dtype=tf.float32)
            qval = model(state_tf)
            action_ = np.argmax(qval.numpy()[0])
            action = action_set[action_]
            test_game.makeMove(action)
            
            state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
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
    print("Training Enhanced Keras DQN in Random Mode...")
    # 增加回合數與 Buffer 以應付隨機模式的挑戰
    model, losses, train_wins = train_agent_keras(epochs=2000, mem_size=2000)
    
    print("Testing Model...")
    win_rate = test_agent(model, test_games=1000)
    print(f"Keras Enhanced DQN Test Win Rate: {win_rate*100:.2f}%")
    
    # Plot losses
    plt.figure(figsize=(10,6))
    if len(losses) > 0:
        plt.plot(moving_average(losses, n=200), label='Enhanced DQN (Keras)', color='blue')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss (Moving Average)')
    plt.title('Keras Enhanced DQN Loss in Random Mode')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_keras_loss.png')
    print("Saved training loss to dqn_keras_loss.png")
