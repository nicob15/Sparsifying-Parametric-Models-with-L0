import torch
import torch.nn as nn
import torch.nn.functional as F
from l0_layer import L0Dense
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class FCNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, h_dim=256):
        super(FCNN, self).__init__()

        self.fc = nn.Linear(input_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, output_dim)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.elu(self.fc(x))
        x = F.elu(self.fc1(x))
        next_obs = self.fc2(x)
        return next_obs


class SparseFCNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, h_dim=256, weight_decay=0., droprate_init=0.5, temperature=2./3.,
                 lambda_coeff=1.):
        super(SparseFCNN, self).__init__()

        self.fc = L0Dense(in_features=input_dim, out_features=h_dim, bias=True, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)
        self.fc1 = L0Dense(in_features=h_dim, out_features=h_dim, bias=True, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)
        self.fc2 = L0Dense(in_features=h_dim, out_features=output_dim, bias=True, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)


    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.elu(self.fc(x))
        x = F.elu(self.fc1(x))
        next_obs = self.fc2(x)
        return next_obs


class L0SINDy_dynamics(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, weight_decay=0., droprate_init=0.5, temperature=2./3.,
                 lambda_coeff=1., degree=3):
        super(L0SINDy_dynamics, self).__init__()

        self.poly = PolynomialFeatures(degree=degree)
        x = np.ones((1, input_dim))
        p = self.poly.fit_transform(x)
        coef_dim = p.shape[1]
        print("policy polynomial of order ", degree)
        print("with {} coefficients".format(coef_dim))
        print(self.poly.get_feature_names_out())

        self.fc = L0Dense(in_features=coef_dim, out_features=1, bias=False, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)
        self.fc1 = L0Dense(in_features=coef_dim, out_features=1, bias=False, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)
        self.fc2 = L0Dense(in_features=coef_dim, out_features=1, bias=False, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        p = torch.from_numpy(self.poly.fit_transform((x).cpu().numpy())).cuda()
        x1 = self.fc(p)
        x2 = self.fc1(p)
        x3 = self.fc2(p)
        next_obs = torch.cat([x1, x2, x3], dim=1)
        return next_obs


class L0SINDy_reward(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, weight_decay=0., droprate_init=0.5, temperature=2. / 3.,
                 lambda_coeff=1., degree=3):
        super(L0SINDy_reward, self).__init__()

        self.poly = PolynomialFeatures(degree=degree)
        x = np.ones((1, input_dim))
        p = self.poly.fit_transform(x)
        coef_dim = p.shape[1]
        print("policy polynomial of order ", degree)
        print("with {} coefficients".format(coef_dim))
        print(self.poly.get_feature_names_out())

        self.fc = L0Dense(in_features=coef_dim, out_features=output_dim, bias=False, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=False)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        p = torch.from_numpy(self.poly.fit_transform((x).cpu().numpy())).cuda()
        r = self.fc(p)
        return r