import gymnasium as gym
from replay_buffer import ReplayBuffer

render = False
if render:
    env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
else:
    env = gym.make('Pendulum-v1', g=9.81)
max_episodes = 1000
max_steps = 200

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
buf_dim = int(max_episodes*max_steps)

seed = 1
training_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=buf_dim)


# create training set
for episode in range(max_episodes):
    observation, info = env.reset(seed=seed)
    for steps in range(max_steps+1):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        training_buffer.store(observation, action, reward, next_observation, done)

        env.render()

        observation = next_observation

        if done:
            done = False
            break

print("Finished creating the training set")

# create test set
max_episodes_test = 100
buf_dim = int(max_episodes*max_steps)

seed = 7
testing_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=buf_dim)

for episode in range(max_episodes_test):
    observation, info = env.reset(seed=seed)
    for steps in range(max_steps + 1):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        testing_buffer.store(observation, action, reward, next_observation, done)

        env.render()

        observation = next_observation

        if done:
            done = False
            break

print("Finished creating the test set")

#env.close()

# learning the dynamics of the pendulum
from models import FCNN, SparseFCNN, L0SINDy_dynamics
from trainer import train_eval_dynamics_model
import torch

h_dim = 64
lr = 3e-4
batch_size = 256
num_epochs = 100

fcnn_model = FCNN(input_dim=obs_dim+act_dim, output_dim=obs_dim, h_dim=h_dim)

if torch.cuda.is_available():
    fcnn_model = fcnn_model.cuda()

optimizer_fcnn = torch.optim.Adam([
    {'params': fcnn_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_fcnn = train_eval_dynamics_model(fcnn_model, optimizer_fcnn, training_buffer, testing_buffer, batch_size, num_epochs)
print("Best testing error FCNN is {} and it was found at epoch {}".format(metrics_fcnn[2], metrics_fcnn[3]))

reg_coefficient = 0.0001
sparsefcnn_model = SparseFCNN(input_dim=obs_dim+act_dim, output_dim=obs_dim, h_dim=h_dim, lambda_coeff=reg_coefficient)

if torch.cuda.is_available():
    sparsefcnn_model = sparsefcnn_model.cuda()

optimizer_sparsefcnn = torch.optim.Adam([
    {'params': sparsefcnn_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_sparsefcnn = train_eval_dynamics_model(sparsefcnn_model, optimizer_sparsefcnn, training_buffer, testing_buffer,
                                               batch_size, num_epochs, l0=True)
print("Best testing error sparse FCNN is {} and it was found at epoch {}".format(metrics_sparsefcnn[2], metrics_sparsefcnn[3]))

degree = 3
reg_coefficient = 0.01
l0sindy_model = L0SINDy_dynamics(input_dim=obs_dim+act_dim, output_dim=obs_dim, degree=degree, lambda_coeff=reg_coefficient)

if torch.cuda.is_available():
    l0sindy_model = l0sindy_model.cuda()

optimizer_fcnn = torch.optim.Adam([
    {'params': l0sindy_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_l0sindy = train_eval_dynamics_model(l0sindy_model, optimizer_fcnn, training_buffer, testing_buffer, batch_size,
                                            num_epochs, l0=True)
print("Best testing error L0 SINDy is {} and it was found at epoch {}".format(metrics_l0sindy[2], metrics_l0sindy[3]))


# creating the plots
import matplotlib.pyplot as plt
import os

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Training and Evaluation Metrics')

data_train = {'FCNN (train)': metrics_fcnn[0], 'SparseFCNN (train)': metrics_sparsefcnn[0], 'L0SINDy (train)': metrics_l0sindy[0]}
methods_train = list(data_train.keys())
values_train = list(data_train.values())

# creating the bar plot
ax1.bar(methods_train, values_train, color='maroon', width=0.4)

data_eval = {'FCNN (eval)': metrics_fcnn[2], 'SparseFCNN (eval)': metrics_sparsefcnn[2], 'L0SINDy (eval)': metrics_l0sindy[2]}
methods_eval = list(data_eval.keys())
values_eval = list(data_eval.values())

ax2.bar(methods_eval, values_eval, color='blue', width=0.4)

save_dir = "figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig.savefig('figures/LearningDynamics.png', dpi=300)

# learning the reward function of the pendulum
from models import L0SINDy_reward
from trainer import train_eval_reward_model

h_dim = 64
lr = 3e-4
batch_size = 256
num_epochs = 50

fcnn_model = FCNN(input_dim=obs_dim+act_dim, output_dim=1, h_dim=h_dim)

if torch.cuda.is_available():
    fcnn_model = fcnn_model.cuda()

optimizer_fcnn = torch.optim.Adam([
    {'params': fcnn_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_fcnn = train_eval_reward_model(fcnn_model, optimizer_fcnn, training_buffer, testing_buffer, batch_size, num_epochs)
print("Best testing error FCNN is {} and it was found at epoch {}".format(metrics_fcnn[2], metrics_fcnn[3]))

reg_coefficient = 0.0001
sparsefcnn_model = SparseFCNN(input_dim=obs_dim+act_dim, output_dim=1, h_dim=h_dim, lambda_coeff=reg_coefficient)

if torch.cuda.is_available():
    sparsefcnn_model = sparsefcnn_model.cuda()

optimizer_sparsefcnn = torch.optim.Adam([
    {'params': sparsefcnn_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_sparsefcnn = train_eval_reward_model(sparsefcnn_model, optimizer_sparsefcnn, training_buffer, testing_buffer,
                                               batch_size, num_epochs, l0=True)
print("Best testing error sparse FCNN is {} and it was found at epoch {}".format(metrics_sparsefcnn[2], metrics_sparsefcnn[3]))

degree = 3
reg_coefficient = 0.01
l0sindy_model = L0SINDy_reward(input_dim=obs_dim+act_dim, output_dim=1, degree=degree, lambda_coeff=reg_coefficient)

if torch.cuda.is_available():
    l0sindy_model = l0sindy_model.cuda()

optimizer_fcnn = torch.optim.Adam([
    {'params': l0sindy_model.parameters()},
], lr=lr, weight_decay=0.0)

metrics_l0sindy = train_eval_reward_model(l0sindy_model, optimizer_fcnn, training_buffer, testing_buffer, batch_size,
                                          num_epochs, l0=True)
print("Best testing error L0 SINDy is {} and it was found at epoch {}".format(metrics_l0sindy[2], metrics_l0sindy[3]))


# creating the plots
import matplotlib.pyplot as plt
import os

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Training and Evaluation Metrics')

data_train = {'FCNN (train)': metrics_fcnn[0], 'SparseFCNN (train)': metrics_sparsefcnn[0], 'L0SINDy (train)': metrics_l0sindy[0]}
methods_train = list(data_train.keys())
values_train = list(data_train.values())

# creating the bar plot
ax1.bar(methods_train, values_train, color='maroon', width=0.4)

data_eval = {'FCNN (eval)': metrics_fcnn[1], 'SparseFCNN (eval)': metrics_sparsefcnn[1], 'L0SINDy (eval)': metrics_l0sindy[0]}
methods_eval = list(data_eval.keys())
values_eval = list(data_eval.values())

ax2.bar(methods_eval, values_eval, color='blue', width=0.4)

save_dir = "figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig.savefig('figures/LearningReward.png', dpi=300)
