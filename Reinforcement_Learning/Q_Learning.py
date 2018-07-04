import gym
from gym import wrappers
import numpy as np

class QAgent(object):
    '''https://en.wikipedia.org/wiki/Q-learning'''
    def __init__(self,env,iter):
        self.env = env
        self.iter = iter
        self.n_states = 40
        self.lr = .1  # Learning rate
        self.min_lr = .001
        self.gamma = 1
        self.eps = .5

        self.best_policy = None

    def softmax(self,logits):
        logits = np.exp(logits)
        return logits/np.sum(logits)


    def obs_to_state(self, obs):
        env_low  = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_dx = (env_high - env_low) / self.n_states
        return np.int32((obs - env_low)/env_dx)

    def run_episode(self,policy=None, outdir = None):
        if policy is None:
            policy = self.best_policy
            self.env = wrappers.Monitor(self.env,directory=outdir,force=True)
        obs = self.env.reset()
        total_reward = 0; step_idx = 0
        while True:
            a,b = self.obs_to_state(obs)
            action = policy[a][b]

            obs, reward, done, _ = self.env.step(action)
            total_reward += (self.gamma**step_idx) * reward
            step_idx += 1
            if done:
                self.env.close()
                break
        return total_reward

    def train(self):
        q_table = np.zeros((self.n_states, self.n_states, 3))
        for i in range(self.iter):
            obs = self.env.reset()
            total_reward = 0
            eta = self.lr #max(self.min_lr, self.initial_lr * (0.9 ** (i/1000)))
            while True:
                a, b = self.obs_to_state(obs)
                if np.random.uniform(0, 1) < self.eps:
                    probs = self.softmax(q_table[a][b])
                    action = np.random.choice(self.env.action_space.n, p=probs)
                else:
                    action = np.argmax(q_table[a][b])


                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                a_, b_ = self.obs_to_state(obs)
                q_table[a][b][action] = q_table[a][b][action] + self.lr * (reward + self.gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
                if done:
                    break
            if i % 100 == 0:
                print('Iteration #{0:d} -- Total reward = {1:.2f}.'.format(i, total_reward))
        self.best_policy = np.argmax(q_table, axis=2)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    outdir = 'Results'
    ## Q learning. Solves iterratively Bellman equation for a Q-function
    agent = QAgent(env,10000)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
