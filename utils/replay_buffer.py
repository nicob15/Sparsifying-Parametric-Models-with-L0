import numpy as np
import torch
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.next_obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=bool)
        self.ptr, self.size, self.max_size = 0, 0, int(size)
    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):

        idxs = np.random.randint(0, self.size, size=batch_size)
        #idxs2 = np.random.randint(0, self.size, size=batch_size)

        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    act=self.acts_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    act=self.acts_buf[:self.size],
                    rew=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size])

    def clear_memory(self):
        print("Emptying memory buffer")
        self.__init__(self.obs_dim, self.act_dim, self.max_size)


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def get_all_samples(self, nr_samples=20000):
        if self.size < nr_samples:
            size = self.size
        else:
            size = nr_samples
        return (
            torch.FloatTensor(self.state[:size]).to(self.device),
            torch.FloatTensor(self.action[:size]).to(self.device),
            torch.FloatTensor(self.next_state[:size]).to(self.device),
            torch.FloatTensor(self.reward[:size]).to(self.device),
            torch.FloatTensor(self.not_done[:size]).to(self.device)
        )

    def clear_memory(self):
        print("Emptying memory buffer")
        self.__init__(self.state_dim, self.action_dim, self.max_size)
