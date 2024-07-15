import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.l0_layer import L0Dense

from utils.agents.feature_libraries.fourier_features import FourierLibrary
from utils.agents.feature_libraries.polynomial_features import PolynomialLibrary
from utils.agents.feature_libraries.generalized_features import GeneralizedLibrary
import os

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, features_dim, max_action, lib, scale=1.0, action_dim=1, droprate=0.5, device='cuda'):
        super(Actor, self).__init__()

        self.actor_coef_1 = L0Dense(in_features=features_dim, out_features=action_dim, bias=False, local_rep=False,
                                    droprate_init=droprate, weight_decay=0.0)
        self.max_action = max_action
        self.lib = lib
        self.scale_features = scale
        self.device = device

        torch.nn.init.normal_(self.actor_coef_1.weights, mean=0, std=0.01)

    def forward(self, x):
        x = x / self.scale_features
        xf = torch.from_numpy(self.lib.transform((x).cpu().numpy())).to(self.device)
        a = self.actor_coef_1(xf)
        a = self.max_action * torch.tanh(a)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=32):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h_dim)
        self.l5 = nn.Linear(h_dim, h_dim)
        self.l6 = nn.Linear(h_dim, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, degree_pi=2, feature_scale=1.0, h_dim=256, tau=0.005,
                 reg_coeff=0.001, droprate=0.5, policy_type='polynomial', frequency=1, device='cuda'):

        self.policy_type = policy_type
        if policy_type == 'polynomial':
            self.lib = PolynomialLibrary(degree=degree_pi, include_bias=True, include_interaction=True)
            x = np.ones((1, state_dim))
            self.lib.fit(x)
            xp = self.lib.transform(x)
            coef_dim = xp.shape[1]
            print("############################")
            print("policy polynomial of degree ", degree_pi)
            print("with {} coefficients".format(coef_dim))
            print(self.lib.get_feature_names())

            self.actor = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                               scale=feature_scale, action_dim=action_dim, device=device).to(device)

            self.actor_target = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                                      scale=feature_scale, action_dim=action_dim, device=device).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())


        if policy_type == 'fourier':
            self.lib = FourierLibrary(n_frequencies=frequency, include_sin=True, include_cos=True, interaction_terms=True)
            x = np.ones((1, state_dim ))
            self.lib.fit(x)
            xf = self.lib.transform(x)
            coef_dim = xf.shape[1]
            print("############################")
            print("fourier policy with frequency ", frequency)
            print("with {} coefficients".format(coef_dim))
            print(self.lib.get_feature_names())

            self.actor = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                               scale=feature_scale, action_dim=action_dim, device=device).to(device)

            self.actor_target = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                                      scale=feature_scale, action_dim=action_dim, device=device).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        if policy_type == "polyfourier":
            poly_lib = PolynomialLibrary(degree=degree_pi, include_bias=True, include_interaction=True)
            fourier_lib = FourierLibrary(n_frequencies=frequency, include_sin=True, include_cos=True, interaction_terms=True)
            self.lib = GeneralizedLibrary([poly_lib, fourier_lib])
            x = np.ones((1, state_dim))
            self.lib.fit(x)
            xpf = self.lib.transform(x)
            coef_dim = xpf.shape[1]
            print("############################")
            print("polynomial with degree {} + fourier policy with frequency {} ".format(degree_pi, frequency))
            print("with {} coefficients".format(coef_dim))
            print(self.lib.get_feature_names())

            self.actor = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                               scale=feature_scale, action_dim=action_dim, device=device).to(device)

            self.actor_target = Actor(features_dim=coef_dim, max_action=max_action, droprate=droprate, lib=self.lib,
                                      scale=feature_scale, action_dim=action_dim, device=device).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())


        self.critic = Critic(state_dim, action_dim, h_dim).to(device)

        self.critic_target = Critic(state_dim, action_dim, h_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.tau = tau
        self.reg_coeff = reg_coeff

        self.device = device

    def select_action(self, state):
        self.actor.eval()
        self.actor.actor_coef_1.training = False
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # regular TD3 is stateless; add this to conform to API
    def reset(self):
        pass

    def print_equations(self, ep):
        self.actor.eval()
        self.actor.actor_coef_1.training = False

        w1 = self.actor.actor_coef_1.weights
        mask1 = self.actor.actor_coef_1.sample_z(1, sample=False)
        _, idx1 = torch.where(mask1 > 0.0)
        w1 = w1[idx1, :]
        mask1 = mask1[:, idx1]
        coef1 = np.array(self.actor.lib.get_feature_names())
        coef1 = coef1[idx1.cpu()]
        print("x0 = s1, x1 = s2, x2 = s3")
        print("pi is equal to:")
        print("coef")
        print(coef1)
        print("mask")
        print(mask1)
        print("weight")
        print(w1.reshape(1, -1))

        with open(os.path.join('figures/' + self.policy_type + '_policy.txt'), 'a') as file:
            file.write("\n")
            file.write("\n")
            file.write("Policy at training episode {}".format(ep))
            file.write("\n")
            file.write("coef {}".format(coef1))
            file.write("\n")
            file.write("mask {}".format(mask1))
            file.write("\n")
            file.write("weight {}".format(w1.reshape(1, -1)))
            file.write("\n")
        file.close()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, policy_noise=0.2, noise_clip=0.3,
              policy_freq=2, clipped_noise=False):

        for it in range(iterations):
            self.critic.train()
            self.actor.train()
            self.actor.actor_coef_1.training = True

            # Sample replay buffer
            x, u, y, r, d = replay_buffer.sample(batch_size)
            state = x
            action = u
            next_state = y
            reward = r
            done = 1.0 - d

            if clipped_noise:
                # Select action according to policy and add clipped noise
                noise = torch.FloatTensor(action.size()).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor_target(next_state).detach()

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                l0_norm_actor = -self.reg_coeff * (self.actor.actor_coef_1.regularization())
                policy_loss = -self.critic.Q1(state, self.actor(state)).mean()
                actor_loss = policy_loss + l0_norm_actor

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor_poly.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic_poly.pth' % (directory, filename))

    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor_poly.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic_poly.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor_poly.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic_poly.pth' % (directory, filename)))

def load(filename, directory):
    if not torch.cuda.is_available():
        return torch.load('%s/%s_all_poly.pth' % (directory, filename), map_location='cpu')
    else:
        return torch.load('%s/%s_all_poly.pth' % (directory, filename))