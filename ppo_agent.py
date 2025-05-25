import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import math
from collections import deque
import os
import time
import csv

# Import the robot environment
from robot_script import RobotEnv, MapType

# Set device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor (Policy) Network with reduced size
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, hidden_dim=32):  # Reduced from 64 to 32
        super(ActorNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.action_bounds = action_bounds  # [low_bounds, high_bounds]
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Mean and log_std for Gaussian policy
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with higher values for better exploration
        self.apply(self._init_weights)
        
        # Initialize the std to be higher at the start for better exploration
        nn.init.constant_(self.log_std_layer.bias, 0.0)  # This gives std of 1.0
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)  # Higher gain for more exploration
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        x = self.feature_extractor(state)
        
        # Output mean and log_std for each action dimension
        action_mean = self.mean_layer(x)
        action_log_std = self.log_std_layer(x)
        
        # Clamp log_std for numerical stability, but ensure minimum exploration
        action_log_std = torch.clamp(action_log_std, -1.2, 2)
        
        return action_mean, action_log_std
    
    def sample_action(self, state, deterministic=False):
        # Get mean and std from the policy network
        state_tensor = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state_tensor)
        std = log_std.exp()
        
        # For deterministic evaluation, just return the mean
        if deterministic:
            action = torch.tanh(mean)
            
            # Rescale to actual action bounds
            action_np = action.cpu().detach().numpy()
            scaled_action = np.zeros_like(action_np)
            
            for i in range(self.action_dim):
                low, high = self.action_bounds[0][i], self.action_bounds[1][i]
                scaled_action[i] = low + (action_np[i] + 1.0) * 0.5 * (high - low)
            
            # Return only the scaled action for deterministic mode
            return scaled_action
        else:
            # Create normal distribution and sample
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)  # Squash to [-1, 1]
            
            # Rescale to actual action bounds
            action_np = action.cpu().detach().numpy()
            scaled_action = np.zeros_like(action_np)
            
            for i in range(self.action_dim):
                low, high = self.action_bounds[0][i], self.action_bounds[1][i]
                scaled_action[i] = low + (action_np[i] + 1.0) * 0.5 * (high - low)
            
            # Calculate log probability for training
            log_prob = normal.log_prob(x_t)
            
            # Apply correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return scaled_action, log_prob.cpu().detach().numpy()

# Critic (Value) Network with reduced size
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):  # Reduced from 64 to 32
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        return self.critic(state)

# Enhanced PPO Agent with improved rewards and exploration
class EnhancedPPOAgent:
    def __init__(self, state_dim, action_dim, action_bounds, 
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, target_kl=0.02,
                 value_coef=0.5, entropy_coef=0.02):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE lambda
        self.clip_ratio = clip_ratio  # PPO clip ratio
        self.target_kl = target_kl  # Target KL divergence for early stopping
        self.value_coef = value_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy bonus coefficient
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, action_bounds).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory buffers for experience collection
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Training statistics
        self.training_step = 0
        self.best_reward = -float('inf')
    
    def select_action(self, state, deterministic=False):
        # Sample action from policy
        if deterministic:
            # For deterministic actions, just return the action directly
            action = self.actor.sample_action(state, deterministic=True)
            return action
        else:
            action, log_prob = self.actor.sample_action(state)
            
            # Get value estimate
            state_tensor = torch.FloatTensor(state).to(device)
            value = self.critic(state_tensor).cpu().detach().numpy()
            
            return action, log_prob, value
        
    def store_transition(self, state, action, log_prob, reward, value, done):
        # Store experience in memory
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def compute_gae(self, last_value, truncated):
        # Compute returns and advantages using GAE (Generalized Advantage Estimation)
        values = np.append(self.values, last_value)
        dones = np.append(self.dones, truncated)
        rewards = np.array(self.rewards)
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def update(self, batch_size=64, n_epochs=10):
        self.training_step += 1
        
        # Get number of experiences
        n_states = len(self.states)
        
        # If no experiences, return
        if n_states == 0:
            return
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions_np = np.array(self.actions)
        actions = torch.FloatTensor(actions_np).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        
        # Compute last value for truncated episode
        if len(self.states) > 0:
            last_state = self.states[-1]
            last_state_tensor = torch.FloatTensor(last_state).to(device)
            last_value = self.critic(last_state_tensor).cpu().detach().numpy()
        else:
            last_value = 0
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(last_value, False)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch training
        for epoch in range(n_epochs):
            # Generate random permutation of indices
            indices = np.random.permutation(n_states)
            
            # Initialize metrics
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            epoch_entropy = 0
            
            # Iterate over mini-batches
            for start_idx in range(0, n_states, batch_size):
                end_idx = min(start_idx + batch_size, n_states)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Un-scale actions for policy evaluation
                unscaled_actions = torch.zeros_like(batch_actions)
                for i in range(self.action_dim):
                    low, high = self.action_bounds[0][i], self.action_bounds[1][i]
                    unscaled_actions[:, i] = 2.0 * (batch_actions[:, i] - low) / (high - low) - 1.0
                
                # Get current action probabilities and values
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                
                # Calculate log probabilities of actions
                x_t = torch.atanh(torch.clamp(unscaled_actions, -0.99, 0.99))  # Inverse of tanh
                new_log_probs = dist.log_prob(x_t)
                
                # Apply correction for tanh squashing
                new_log_probs -= torch.log(1 - unscaled_actions.pow(2) + 1e-6)
                new_log_probs = new_log_probs.sum(dim=1, keepdim=True)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Calculate value estimate
                values = self.critic(batch_states)
                
                # Calculate ratio for importance sampling
                ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # PPO actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Accumulate metrics
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += value_loss.item()
                epoch_entropy += entropy.item()
                
                # Check for early stopping with KL divergence
                approx_kl = (batch_log_probs - new_log_probs).mean().item()
                if approx_kl > self.target_kl:
                    break
            
            # Print metrics for last epoch
            if epoch == n_epochs - 1:
                num_batches = (n_states + batch_size - 1) // batch_size
                print(f"Update {self.training_step}, Epoch {epoch+1}/{n_epochs} - Actor Loss: {epoch_actor_loss/num_batches:.4f}, Critic Loss: {epoch_critic_loss/num_batches:.4f}, Entropy: {epoch_entropy/num_batches:.4f}")
        
        # Clear memory after update
        self.clear_memory()
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'best_reward': self.best_reward
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            if 'training_step' in checkpoint:
                self.training_step = checkpoint['training_step']
            if 'best_reward' in checkpoint:
                self.best_reward = checkpoint['best_reward']
            print(f"Model loaded from {path}")
            return True
        return False

# Enhanced rewards wrapper for the robot environment
class EnhancedRobotEnv:
    def __init__(self, env):
        self.env = env
        self.previous_distance_to_goal = None
        self.previous_action = None
        self.step_count = 0
        self.prev_pos = None
        self.total_distance_moved = 0
        
    def reset(self):
        state, info = self.env.reset()
        
        # Get initial distance to goal
        goal_x, goal_y, goal_width, goal_height = self.env.goal  # <-- FIXED
        robot_x, robot_y = self.env.robot.x, self.env.robot.y
        self.previous_distance_to_goal = np.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)
        
        self.previous_action = np.array([0.0, 0.0])
        self.step_count = 0
        self.prev_pos = (robot_x, robot_y)
        self.total_distance_moved = 0
        
        return state, info
    
    def step(self, action):
        # Take action in the environment
        next_state, done, truncated, info = self.env.step(action)
        
        # Extract current state components
        left_sensor, front_sensor, right_sensor, color_sensor, v, w = next_state
        robot_x, robot_y = self.env.robot.x, self.env.robot.y
        
        # Calculate enhanced reward
        enhanced_reward = self._calculate_enhanced_reward(
            robot_x, robot_y, 
            left_sensor, front_sensor, right_sensor, 
            color_sensor, v, w, done, info
        )
        
        # Update previous position and action
        self.prev_pos = (robot_x, robot_y)
        self.previous_action = action
        self.step_count += 1
        
        return next_state, enhanced_reward, done, truncated, info
    
    def _calculate_enhanced_reward(self, robot_x, robot_y, left_sensor, front_sensor, right_sensor, 
                                  color_sensor, v, w, done, info):
        # Base reward
        reward = 0.0
        
        if color_sensor == 1:
            reward += 1.0
            return reward
        if done and color_sensor==0:
            reward -= 1
            # if info.get('timeout', False):
            #     reward -= 0.5
            # return reward
        return reward
        
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

# Training function
def train_ppo(env, agent, n_steps=2048, max_episodes=500, update_every=2048, render=False, 
              save_path="ppo_robot_models", continue_training=False):
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    avg_rewards = []
    step_count = 0
    episode_count = 0
    best_avg_reward = agent.best_reward if hasattr(agent, 'best_reward') else -float('inf')
    
    # Load best model if continuing training
    if continue_training:
        best_model_path = os.path.join(save_path, "best_ppo_robot.pt")
        if os.path.exists(best_model_path):
            agent.load(best_model_path)
    
    # Reset environment
    state, _ = env.reset()
    episode_reward = 0
    episode_step_count = 0
    
    # Start time for logging
    start_time = time.time()
    last_save_time = start_time
    
    episode_data = []  # To store data for each episode
    success_count = 0
    crash_count = 0
    timeout_count = 0
    # Training loop
    while episode_count < max_episodes:
        
        action, log_prob, value = agent.select_action(state) # Select action
        next_state, reward, done, truncated, _ = env.step(action) # Take action in environment
        agent.store_transition(state, action, log_prob, reward, value, done) # Store transition
        
        # Update state and metrics
        state = next_state
        episode_reward += reward
        step_count += 1
        episode_step_count += 1
        
        # Render if enabled
        if render and episode_count % 20 == 0:  # Render every 20 episodes
            env.render()
        
        # Episode ended or update timing
        if done or truncated or step_count % update_every == 0:
            if done or truncated:
                # Determine result type
                info_type = "timeout"
                # If your env returns info, use it to check for success/collision
                # Here, you may need to adapt this part to your info dict
                # For example:
                # info = ... (get info from env if available)
                # For now, let's assume success if reward > 0.5, crash if reward < -0.5
                if episode_reward > 0.5:
                    info_type = "success"
                    success_count += 1
                elif episode_reward < -0.5:
                    info_type = "crash"
                    crash_count += 1
                else:
                    timeout_count += 1

                avg_window = min(100, len(episode_rewards))
                avg_reward = np.mean(episode_rewards[-avg_window:]) if episode_rewards else 0

                episode_data.append({
                    "episode": episode_count,
                    "reward": episode_reward,
                    "length": episode_step_count,
                    "result": info_type,
                    "success_rate": success_count / (episode_count + 1),
                    "crash_rate": crash_count / (episode_count + 1),
                    "timeout_rate": timeout_count / (episode_count + 1),
                    "avg_reward_100": avg_reward
                })




                state, _ = env.reset() # Reset environment
                episode_rewards.append(episode_reward) # Record episode reward
                elapsed_time = time.time() - start_time # Calculate time elapsed
                print(f"Episode {episode_count+1}, Reward: {episode_reward:.2f}, Steps: {step_count}, Time: {elapsed_time:.2f}s") # Log episode results
                
                episode_reward = 0
                episode_count += 1
                episode_step_count = 0
                
                # Calculate average reward over last 100 episodes
                if len(episode_rewards) > 0:
                    avg_window = min(100, len(episode_rewards))
                    avg_reward = np.mean(episode_rewards[-avg_window:])
                    avg_rewards.append(avg_reward)
                    
                    # Save best model
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        agent.best_reward = best_avg_reward
                        agent.save(os.path.join(save_path, "best_ppo_robot.pt"))
                        print(f"New best average reward: {best_avg_reward:.2f}")
                    
                    # Log average reward every 100 episodes
                    if episode_count % 100 == 0:
                        print(f"Average Reward (last {avg_window} eps): {avg_reward:.2f}, Best: {best_avg_reward:.2f}")
            
            # Update policy if enough steps
            if step_count % update_every == 0:
                print(f"\nUpdating policy after {step_count} steps...")
                agent.update(batch_size=64, n_epochs=10)
            
            # Periodic saving (every 5 minutes)
            current_time = time.time()
            if current_time - last_save_time > 300:  # 300 seconds = 5 minutes
                agent.save(os.path.join(save_path, "latest_ppo_robot.pt"))
                last_save_time = current_time
                
                # Plot training curves
                if len(episode_rewards) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(episode_rewards)
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Training Rewards')
                    plt.savefig(os.path.join(save_path, 'training_rewards.png'))
                    plt.close()
                
                if len(avg_rewards) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(avg_rewards)
                    plt.xlabel('Episode')
                    plt.ylabel('Average Reward (50 episodes)')
                    plt.title('Average Training Rewards')
                    plt.savefig(os.path.join(save_path, 'average_rewards.png'))
                    plt.close()
    
        # --- Write to CSV at the end ---
    
    
    keys = ["episode", "reward", "length", "result", "success_rate", "crash_rate", "timeout_rate", "avg_reward_100"]
    with open(os.path.join(save_path, "training_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(episode_data)

    # Save final model
    agent.save(os.path.join(save_path, "final_ppo_robot.pt"))
    
    # Final plots
    if len(episode_rewards) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.savefig(os.path.join(save_path, 'final_training_rewards.png'))
        plt.close()
    
    if len(avg_rewards) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (50 episodes)')
        plt.title('Average Training Rewards')
        plt.savefig(os.path.join(save_path, 'final_average_rewards.png'))
        plt.close()
    
    return episode_rewards, avg_rewards

# Evaluation function
def evaluate_ppo(env, agent, n_episodes=1000, render=False):
    total_rewards = []
    success_count = 0
    crash_count = 0
    timeout_count = 0
    
    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        max_steps = 500  # Limit evaluation episode length
        
        while not (done or truncated) and step_count < max_steps:
            # Select action (deterministic for evaluation)
            action = agent.select_action(state, deterministic=True)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Render if enabled
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        
        # Check outcome
        if info['at_goal']:
            success_count += 1
            # print(f"Evaluation Episode {i+1}: SUCCESS! Reward = {episode_reward:.2f}, Steps = {step_count}")
        elif info['collision']:
            crash_count +=  1
            # print(f"Evaluation Episode {i+1}: COLLISION. Reward = {episode_reward:.2f}, Steps = {step_count}")
        else:
            timeout_count += 1
            # print(f"Evaluation Episode {i+1}: TIMEOUT. Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / n_episodes * 100
    crash_rate = crash_count / n_episodes *100
    timeout_rate = timeout_count / n_episodes * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
    print(f"Crash Rate: {crash_rate:.1f}% ({crash_count}/{n_episodes})")
    print(f"Timeout Rate: {timeout_rate:.1f}% ({timeout_count}/{n_episodes})")
    
    return total_rewards, avg_reward, success_rate


# Main function
def main():
    # Create training environment (straight line)
    env_train_base = RobotEnv(map_type=MapType.STRAIGHT)
    env_train = EnhancedRobotEnv(env_train_base)
    
    # Get state and action dimensions
    state_dim = env_train_base.observation_space.shape[0]
    action_dim = env_train_base.action_space.shape[0]
    action_bounds = [env_train_base.action_space.low, env_train_base.action_space.high]
    
    # Create enhanced PPO agent with smaller networks
    agent = EnhancedPPOAgent(
        state_dim, 
        action_dim, 
        action_bounds,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.01,  # Reduced KL divergence target for more stable learning
        value_coef=0.5,
        entropy_coef=0.01  # Reduced entropy coefficient
    )
    
    # Train agent
    train_ppo(
        env_train, 
        agent, 
        n_steps=1024,         # Smaller batch size
        max_episodes=100000,     # Maximum number of episodes
        update_every=1024,    # Update policy more frequently
        render=False,          # Enable rendering
        save_path="ppo_robot_models",
        continue_training=True
    )
    
    # Load best model for evaluation
    best_model_path = os.path.join("ppo_robot_models", "best_ppo_robot.pt")
    agent.load(best_model_path)
    
    # Evaluate on training environment
    print("\nEvaluating on training environment:")
    eval_rewards_train, avg_reward_train, success_rate_train = evaluate_ppo(env_train, agent)
    

    # Evaluate on Narrow environment
    print("\nEvaluating on Narrow:")
    env_eval_base = RobotEnv(map_type=MapType.NARROW)
    env_eval = EnhancedRobotEnv(env_eval_base)

    print("\nEvaluating on straight line environment:")
    eval_rewards_narrow, avg_reward_narrow, success_rate_narrow = evaluate_ppo(env_eval, agent)
    
    # Create evaluation environment (T-junction)
    env_eval_base = RobotEnv(map_type=MapType.T_JUNCTION)
    env_eval = EnhancedRobotEnv(env_eval_base)
    
    # Evaluate on T-junction environment
    print("\nEvaluating on T-junction environment:")
    eval_rewards_tjunction, avg_reward_tjunction, success_rate_tjunction = evaluate_ppo(env_eval, agent)
    
    # Close environments
    env_train.close()
    env_eval.close()
    
    # Print final results
    print("\nFinal Results:")
    print(f"Training Environment (Narrow Line) - Average Reward: {avg_reward_train:.2f}, Success Rate: {success_rate_train:.1f}%")
    print(f"Evaluation Environment (Narrow) - Average Reward: {avg_reward_narrow:.2f}, Success Rate: {success_rate_narrow:.1f}%")
    print(f"Evaluation Environment (T-Junction) - Average Reward: {avg_reward_tjunction:.2f}, Success Rate: {success_rate_tjunction:.1f}%")

if __name__ == "__main__":
    main()
