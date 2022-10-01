import torch
import torch.nn as nn
import numpy as np
import gym
import sklearn
import sklearn.preprocessing
from actor import Actor
from critic import Critic


NUM_EPISODES = 300
GAMMA = 0.99  # discount factor


def get_action(dist_parameters):
    mu = dist_parameters[0]
    sigma = dist_parameters[0]

    return torch.normal(mu, sigma)


def get_scaler(env):
    state_space_samples = np.array(
        [env.observation_space.sample() for _ in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    return scaler


def TD_advantage_actor_critic():
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("Pendulum-v1")
    # torch.autograd.set_detect_anomaly(True)

    actor = Actor(input_dim=env.observation_space.shape[0], n_hidden1=40, n_hidden2=40, output_dim=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001) # 

    critic = Critic(input_dim=env.observation_space.shape[0], n_hidden1=400, n_hidden2=400, output_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.01) # ,
    critic_loss_func = torch.nn.MSELoss()
    # critic_loss_func = torch.nn.L1Loss()

    softplus = nn.Softplus()

    scaler = get_scaler(env)

    episode_history = []

    for ep in range(NUM_EPISODES):
        curr_state = env.reset()
        cumulative_reward = 0 
        steps = 0
        done = False
        while not done:
            # actor returns parameters of a Normal distribution
            dist_parameters = actor(torch.tensor(curr_state))
            mu = dist_parameters[0]
            sigma = softplus(dist_parameters[1]) + 1e-5
            N = torch.distributions.normal.Normal(mu, sigma)

            action = N.sample().unsqueeze(dim=0)
            action = torch.clip(action, min=env.action_space.low[0], max=env.action_space.high[0])

            next_state, reward, done, _ = env.step(action)
            next_state = scaler.transform([next_state]).astype(np.float32).squeeze(0)
            steps += 1
            cumulative_reward += reward

            # update critic
            td_target = reward + GAMMA*critic(torch.tensor(next_state))
            curr_prediction = critic(torch.tensor(curr_state))
            advantage = td_target.item() - curr_prediction.item()
            critic_loss = critic_loss_func(curr_prediction, td_target)
            # print(f"critic loss: {critic_loss}")
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # update actor
            actor_loss = -torch.log(N.cdf(action) + 1e-5)*(advantage)
            # print(f"actor loss: {actor_loss}")
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            curr_state = next_state
        episode_history.append(cumulative_reward)
        print(f"Episode #{ep}:")
        print(f"\t - Steps: {steps}")
        print(f"\t - Reward: {cumulative_reward}")

        if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
            print("****************Solved***************")
            print("Mean cumulative reward over last 100 episodes:", 
                  np.mean(episode_history[-100:]))
    
    np.save("rewards.npy", episode_history)


if __name__ == "__main__":
    TD_advantage_actor_critic()