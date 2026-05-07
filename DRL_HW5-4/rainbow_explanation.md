# Rainbow DQN 原理與實作解析報告

在 `DRL_HW5-4` 中，我們挑戰了深度強化學習 (DRL) 中公認的集大成之作：**Rainbow DQN**。Rainbow 演算法並不是一種全新的神經網路架構，而是巧妙地整合了六大 DQN 的改良技術。在面對複雜度極高、每次開局位置都不同的 `random` 模式 GridWorld 時，Rainbow DQN 展現出了極高的樣本效率 (Sample Efficiency) 與收斂穩定性。

本報告將為您拆解 `rainbow_dqn.py` 程式碼中實作的全套六大核心技術。

---

## 1. 雙重 Q 學習 (Double DQN, DDQN)
- **實作位置**：`train_rainbow()` 迴圈中的 Target 分佈運算。
- **原理解析**：傳統 DQN 在計算 TD Target 時會直接取 $\max Q$，容易造成「過度估計」。我們讓 Online Net 負責選擇最佳動作 (`next_actions = online_net.act(...)`)，再交由 Target Net 來輸出該動作對應的機率分佈 (`target_net(...)`)。這有效地抑制了神經網路對未知的盲目樂觀。

## 2. 競爭網路架構 (Dueling Network Architecture)
- **實作位置**：`RainbowNet` 類別的 `forward` 方法。
- **原理解析**：我們在網路最後一層前將輸出拆分成兩個分支：「狀態價值 $V(s)$」與「優勢函數 $A(s, a)$」。
  > $Q(S, a) = V(S) + \left( A(S, a) - \text{mean}(A) \right)$
  在 GridWorld 中，有些狀態下無論往哪裡走都不會立刻有結果，這時網路只需要更新 $V(S)$ 即可，不需花費資源去精準微調每個動作的 $A(s, a)$，大大提升了收斂效率。

## 3. 多步學習 (Multi-step Learning / N-step Returns)
- **實作位置**：`train_rainbow()` 中的 `n_step_buffer`。
- **原理解析**：不同於一般 DQN 每次只看「走一步後的 Reward (1-step)」，我們使用長度為 3 的 `deque`，讓 Agent 一次看未來 3 步的累積回報。這加速了遠處獎勵（如方格世界邊緣的寶藏）傳遞回起點的速度，讓 Agent 能夠更快學到長遠的策略。

## 4. 雜訊網路 (Noisy Nets)
- **實作位置**：`NoisyLinear` 類別。
- **原理解析**：傳統的 $\epsilon$-greedy 探索方式太過呆板（純隨機亂走）。我們將神經網路的全連接層換成了 `NoisyLinear`，為網路權重加入了**可學習的高斯雜訊**。
  這讓模型具備「自主探索」的能力：遇到不熟悉的狀態時，雜訊會促使模型嘗試不同動作；當模型確信某條路徑是正確的，它會自動將雜訊的權重 ($\sigma$) 降到最低，完美解決了探索與利用 (Exploration vs Exploitation) 的兩難。

## 5. 優先經驗回放 (Prioritized Experience Replay, PER)
- **實作位置**：`SumTree` 與 `PrioritizedReplayBuffer` 類別。
- **原理解析**：在記憶體中，不是每一筆經驗都同樣重要。Agent「意外撞到牆壁」或「突然吃到寶藏」的經驗，比「在空地上平安無事走一步」更具學習價值。
  我們使用 **SumTree** 資料結構，根據 TD Error (預測誤差) 作為權重來抽樣記憶。誤差越大的經驗被抽出的機率越高，讓模型將有限的訓練次數專注在「自己還不熟悉」的關卡上。同時搭配 Importance Sampling (IS) Weights，修正非均勻抽樣帶來的偏差。

## 6. 分佈式強化學習 (Distributional RL / Categorical DQN - C51)
- **實作位置**：`projection_distribution()` 與 Loss 計算。
- **原理解析**：這是整份程式碼中最複雜但也最關鍵的一環。傳統 DQN 只預測 Q 值的「平均期望值」(Scalar)，但現實中同樣的動作可能引發截然不同的後果。
  我們將原本單一的 Q 值，轉換成一個包含 51 個離散原子 (Atoms) 的**機率分佈**（從 Vmin 到 Vmax）。在計算 TD Error 時，我們投影下一個狀態的機率分佈，並使用**交叉熵損失 (Cross-Entropy Loss)** 來拉近預測分佈與目標分佈的距離。這提供了極其豐富的學習訊號，極大化了神經網路學習環境動態特性的能力。

---

## 結語

在這份 `DRL_HW5-4` 的實作中，我們將這六個看起來毫不相干的改進技術組裝成了 **Rainbow DQN**。
當您在背景執行 `rainbow_dqn.py` 時，可以發現即使面對每次開局地圖全換的 `random` 模式，Rainbow DQN 依然能以極高的效率在幾千步之內就抓到破解 GridWorld 的訣竅，並且訓練過程不再像基礎 DQN 那樣劇烈震盪，這正是集成了所有演算法優點的強大威力！
