# Deep Q-Network for Flappy Bird

This project implements an autonomous agent capable of mastering the Flappy Bird environment using raw pixel data. The solution utilizes a Deep Q-Network (DQN) integrated with a Convolutional Neural Network (CNN) to process stacked, binary-thresholded frames.

## Project Workflow

The system operates in a closed-loop cycle to infer spatial structure and temporal dynamics directly from image frames.

| Phase | Description |
| :--- | :--- |
| **Observation** |Capture raw RGB frames from the FlappyBird-v0 environment. |
| **Preprocessing** |Simplify visual data through cropping, thresholding, and normalization. |
| **Action Selection** | Select actions (Wait or Flap) based on an $\epsilon$-greedy policy. |
| **Experience Replay** | Store transitions in a 50,000-size buffer to reduce sample correlation. |
| **Optimization** | Update the Q-Network using Mean Squared Error (MSE) loss against target values. |

## Image Preprocessing Pipeline

Observations are processed to focus the agent on task-relevant geometry and motion.

| Stage | Operation | Details                                                                     |
| :--- | :--- |:----------------------------------------------------------------------------|
| **Spatial Cropping** | Background Removal | The lower 110 pixels (ground) are removed to isolate obstacles.             |
| **Binary Thresholding** | Contrast Enhancement | Frames are converted to binary using the object outline color (84, 56, 71). |
| **Rescaling** | Dimensionality Reduction | Images are resized to $84 \times 84$ and normalized to the range [0, 1].    |
| **Temporal Stacking** | Motion Perception | 4 consecutive frames are stacked to form a (4, 84, 84) state tensor.        |



## Neural Network Architecture

The model uses a CNN designed for efficient spatial feature extraction from the processed state.

| Layer | Type | Configuration | Activation |
| :--- | :--- | :--- | :--- |
| **Layer 1** | Convolutional | 32 filters, $7 \times 7$ kernel, Stride 3 | ReLU6|
| **Layer 2** | Convolutional | 64 filters, $5 \times 5$ kernel, Stride 2 | ReLU6 |
| **Layer 3** | Convolutional | 64 filters, $3 \times 3$ kernel, Stride 1 | ReLU6
| **Layer 4** | Fully Connected | 256 Hidden Units |ReLU6|
| **Output** | Linear | 2 Output Units (Wait, Flap) | None


## Stabilized Q-Learning (Target Networks)

To ensure training stability and prevent divergence, the implementation utilizes a target network mechanism.

* **Dual Network System**: A separate target network is maintained to compute Q-value targets.
* **Decoupled Updates**: The target network weights are synchronized with the primary Q-network every 20 episodes.
* **Oscillation Mitigation**: This approach mitigates the moving target problem during the optimization process.

## Training Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Gamma ($\gamma$)** | 0.99  |
| **Learning Rate ($\alpha$)** |$1 \times 10^{-4}$ |
| **Batch Size** | 32  |
| **Replay Buffer** | 50,000  |
| **Exploration ($\epsilon$)** |$1.0 \rightarrow 1 \times 10^{-4}$ over 10,000 steps |
| **Frame Skipping** | 2 frames per action|

## Evaluation Performance

| Metric | Result |
| :--- | :--- |
| **Peak Score** |642 pipes  |
| **Average Score** | 115.5  |
| **Top 5 Scores** | 642, 348, 326, 324, 321 pipes  |

## Project Structure

* `qnetwork.py`: PyTorch implementation of the CNN architecture.
* `humanrendering.py`: Wrapper for displaying `rgb_array` frames via Pygame.
* `train-model.py`: Main training loop with experience replay and target updates.
* `test-model.py`: Script for greedy policy evaluation using trained weights.