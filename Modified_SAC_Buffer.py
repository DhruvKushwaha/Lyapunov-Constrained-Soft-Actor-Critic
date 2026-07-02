# -----------------------------------------------------------------------------------
#                   Storage
# -----------------------------------------------------------------------------------
"""
Replay buffer for **LC-SAC** with an extra ``X_error`` channel (tracking error).

``x_error_dim`` controls the width of the X_error field: 6 for 2D quadrotor,
4 for cartpole, 12 for 3D quadrotor.  Passed to ``__init__`` at construction time.

For vanilla SAC, use ``safe_control_gym``'s ``SACBuffer`` via ``rl.train`` / ``SAC``.
"""
import numpy as np
import torch
from gymnasium.spaces import Box

class SACBuffer(object):
    """Fixed-size replay storage with ``obs``, ``act``, ``X_error``, etc., for LC-SAC training."""

    # Attributes: max_size, batch_size, scheme, keys — see __init__ / reset.

    def __init__(self, obs_space, act_space, max_size, batch_size=None, x_error_dim=6):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size

        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n

        N = max_size

        self.scheme = {
            'obs': {
                'vshape': (N, *obs_dim)
            },
            'next_obs': {
                'vshape': (N, *obs_dim)
            },
            'act': {
                'vshape': (N, act_dim)
            },
            'rew': {
                'vshape': (N, 1)
            },
            'mask': {
                'vshape': (N, 1),
                'init': np.ones
            },
            'X_error': {
                'vshape': (N, x_error_dim)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocate space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape).astype(dtype)

        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        '''Returns current size of the buffer.'''
        return self.buffer_size

    def state_dict(self):
        '''Returns a snapshot of current buffer.'''
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        '''Restores buffer from previous state.'''
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        # batch size
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            v_ = np.asarray(v, dtype=dtype).reshape((n,) + shape)

            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos:self.pos + n] = v_
            else:
                # wrap around
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos:self.max_size] = v_[:-remain_n]
                self.__dict__[k][:remain_n] = v_[-remain_n:]

        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self, batch_size=None, device=None):
        '''Returns data batch.'''
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][1:]
            v = self.__dict__[k].reshape(-1, *shape)[indices]
            if device is None:
                batch[k] = torch.as_tensor(v)
            else:
                batch[k] = torch.as_tensor(v, device=device)
        return batch