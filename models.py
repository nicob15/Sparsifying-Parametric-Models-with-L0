import torch
import torch.nn as nn
import torch.nn.functional as F
from l0_layer import L0Dense
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from torch.nn import init


class FCNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, h_dim=256, use_bias=True):
        super(FCNN, self).__init__()

        self.use_bias = use_bias
        self.fc = nn.Linear(input_dim, h_dim, bias=use_bias)
        self.fc1 = nn.Linear(h_dim, h_dim, bias=use_bias)
        self.fc2 = nn.Linear(h_dim, output_dim, bias=use_bias)

        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_normal_(self.fc.weight, mode='fan_out')
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')

        if self.use_bias:
            self.fc.bias.data.fill_(0)
            self.fc1.bias.data.fill_(0)
            self.fc2.bias.data.fill_(0)
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
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=True)
        self.fc1 = L0Dense(in_features=h_dim, out_features=h_dim, bias=True, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=True)
        self.fc2 = L0Dense(in_features=h_dim, out_features=output_dim, bias=True, weight_decay=weight_decay,
                          droprate_init=droprate_init, temperature=temperature, lamba=lambda_coeff, local_rep=True)
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

    def print_equations(self):
        w1 = self.fc.weights
        mask1 = self.fc.sample_z(1, sample=False)
        _, idx1 = torch.where(mask1 > 0.0)
        w1 = w1[idx1, :]
        mask1 = mask1[:, idx1]
        coef1 = self.poly.get_feature_names_out()
        coef1 = coef1[idx1.cpu()]
        print("x0 = s1, x1 = s2, x2 = s3, x3 = u")
        print("s1 is equal to:")
        print(coef1)
        print(mask1)
        print(w1.reshape(1, -1))

        w2 = self.fc1.weights
        mask2 = self.fc1.sample_z(1, sample=False)
        _, idx2 = torch.where(mask2 > 0.0)
        w2 = w2[idx2, :]
        mask2 = mask2[:, idx2]
        coef2 = self.poly.get_feature_names_out()
        coef2 = coef2[idx2.cpu()]
        print("x0 = s1, x1 = s2, x2 = s3, x3 = u")
        print("s2 is equal to:")
        print(coef2)
        print(mask2)
        print(w2.reshape(1, -1))

        w3 = self.fc2.weights
        mask3 = self.fc2.sample_z(1, sample=False)
        _, idx3 = torch.where(mask3 > 0.0)
        w3 = w3[idx3, :]
        mask3 = mask3[:, idx3]
        coef3 = self.poly.get_feature_names_out()
        coef3 = coef3[idx3.cpu()]
        print("x0 = s1, x1 = s2, x2 = s3, x3 = u")
        print("s3 is equal to:")
        print(coef3)
        print(mask3)
        print(w3.reshape(1, -1))


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

    def print_equations(self):
        w1 = self.fc.weights
        mask1 = self.fc.sample_z(1, sample=False)
        _, idx1 = torch.where(mask1 > 0.0)
        w1 = w1[idx1, :]
        mask1 = mask1[:, idx1]
        coef1 = self.poly.get_feature_names_out()
        coef1 = coef1[idx1.cpu()]
        print("x0 = s1, x1 = s2, x2 = s3, x3 = u")
        print("r is equal to:")
        print(coef1)
        print(mask1)
        print(w1.reshape(1, -1))