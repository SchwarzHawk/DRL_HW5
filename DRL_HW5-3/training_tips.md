# DQN 針對 Random 模式的訓練強化技巧 (Keras 實作)

在我們將模型框架從 PyTorch 轉換為 **Keras / TensorFlow** 的同時，我們也為模型引入了幾個強化的「訓練技巧 (Training Tips)」。這些技巧特別針對了難度極高的 **`random` 模式**進行設計。

## 為什麼 Random 模式特別難訓練？

在 `player` 或 `static` 模式下，陷阱與目標物的位置是固定的，AI 只需要「背下」地圖的一條最佳路徑即可獲勝。但在 `random` 模式下，每次開局時玩家、陷阱、寶藏、牆壁的位置都完全隨機。這要求神經網路不能只是死背地圖，而必須真正學會**理解畫面中不同物件的相對位置關係與規則**。這也使得 Q 值的波動變得極為劇烈，若缺乏穩定的機制，模型很容易崩潰 (Loss 爆炸) 或陷入區域最佳解。

為此，我們在 `dqn_keras.py` 中加入了以下三項訓練技巧來穩定收斂：

---

### 1. 梯度裁剪 (Gradient Clipping)
**實作方式**：在 Keras 的 Optimizer 參數中加入 `clipnorm=1.0`。
```python
optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
```
**原理與影響**：
在 `random` 模式中，如果 Agent 剛好重生在陷阱旁邊立刻死掉，或是意外走到目標，計算出來的 TD Error 往往會瞬間變得非常大。這會產生過大的梯度 (Gradient)，使得神經網路的權重瞬間被推向極端值，破壞前面辛苦學到的特徵。
引入 **Gradient Clipping** 會限制梯度的最大長度 (Norm)，強制規定「即便你發現了一個天大的驚喜/驚嚇，你每次修正的幅度也只能這麼大」。這能有效防止模型崩潰，讓 Loss 收斂得更加平滑。

### 2. 學習率排程衰減 (Learning Rate Scheduling)
**實作方式**：使用 `tf.keras.optimizers.schedules.ExponentialDecay` 取代固定的學習率。
```python
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True
)
```
**原理與影響**：
強化學習在訓練前期需要快速大幅度更新權重以確立大方向（例如「靠近 + 號是好的」）；但在訓練後期，為了精細區分邊界條件（例如「牆壁旁邊的陷阱要怎麼繞」），過大的學習率反而會讓模型在最佳解附近不斷震盪而無法收斂。
我們透過**指數衰減 (Exponential Decay)**，讓學習率隨著訓練步數慢慢變小。這幫助了 Keras 網路在後期能夠穩定地微調 Q 值，進而提升最終測試的勝率表現。

### 3. 探索機率平滑衰減 (Smoothed Epsilon Decay)
**實作方式**：調整 Epsilon 衰減率為 `0.998`，並將下限 (`epsilon_min`) 設為 `0.05`。
```python
epsilon_decay = 0.998
epsilon_min = 0.05
```
**原理與影響**：
傳統的衰減率（例如 `0.99` 或 `0.9`）降得太快，導致 AI 在還沒有看過足夠多種 `random` 地圖配置前，就放棄了隨機探索，開始完全依賴自己半生不熟的網路去決策，最終導致學到錯誤的刻板印象。
放慢衰減速度 (`0.998`) 可以確保 Agent 能夠有更長的「探索期」來遊歷隨機生成的地圖，並收集更多元的經驗放進 Replay Buffer 中；而保留 `5%` 的探索下限則能讓模型持續保持隨機應變的彈性。

---

## 總結
透過將 **Gradient Clipping**、**LR Scheduling** 與 **適當的 Epsilon 控制** 整合進 Keras 的 DQN 模型中，我們不僅完成了框架的轉換，更賦予了模型對抗高隨機性環境的穩定能力。您現在可以執行 `dqn_keras.py`，並觀看輸出的 `dqn_keras_loss.png`，體會這些訓練技巧所帶來的平穩收斂效果。
