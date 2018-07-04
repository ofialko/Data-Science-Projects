import numpy as np
import gym

class PolicyAgent(object):
    def __init__(self,env,max_iter):
        self.gamma = 1
        self.env = env
        self.n = 100
        self.max_iter = max_iter
        self.eps = 1e-10
        self.best_policy = None

    def run_episode(self,policy=None,render = False):
        if policy is None:
            policy = self.best_policy
            render = True
        obs = self.env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            obs, reward, done, _ = self.env.step(int(policy[obs]))
            total_reward += (self.gamma ** step_idx * reward)
            step_idx += 1
            if render:
                self.env.render()
            if done:
                self.env.close()
                break
        return total_reward

    def evaluate_policy(self,env, policy):
        """ Evaluates a policy by running it n times."""
        scores = [self.run_episode(policy) for _ in range(self.n)]
        return np.mean(scores)

    def extract_policy(self,value_fun):
        """ Extract the policy given a value-function """
        policy = np.zeros(self.env.env.nS)
        for s in range(self.env.env.nS):
            q_sa = np.zeros(self.env.env.nA)
            for a in range(self.env.env.nA):
                for next_sr in self.env.env.P[s][a]:
                    p, s_, r, _ = next_sr
                    q_sa[a] += (p * (r + self.gamma * value_fun[s_]))
            policy[s] = np.argmax(q_sa)
        return policy

    def compute_value_fun(self,policy):
        value_fun = np.zeros(self.env.env.nS)
        while True:
            prev_v = np.copy(value_fun)
            for s in range(self.env.env.nS):
                policy_a = policy[s]
                value_fun[s] = sum([p * (r + self.gamma * prev_v[s_]) for p, s_, r, _ in self.env.env.P[s][policy_a]])
            if (np.sum((np.fabs(prev_v - value_fun))) <= self.eps):
                break
        return value_fun

    def train(self):
        """ Policy-iteration algorithm """
        policy = np.random.choice(self.env.env.nA,size=(self.env.env.nS))

        for i in range(self.max_iter):
            value_fun = self.compute_value_fun(policy)
            new_policy = self.extract_policy(value_fun)
            if (np.all(policy == new_policy)):
                break
            policy = new_policy
        self.best_policy = policy


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')

    agent = PolicyAgent(env,10000)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(render=True)
    env.close()
    print('Policy average score = ', reward)
