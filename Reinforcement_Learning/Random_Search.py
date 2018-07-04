import gym
from gym import wrappers
import numpy as np

class RandomAgent(object):
    '''Generates a set of random solutions
       and chooses the one with the highest reward.'''
    def __init__(self,env,n_policy):
        self.env = env
        self.n_policy = n_policy

        self.x = env.observation_space.shape[0]
        self.y = env.action_space.n
        self.best_policy = None

    def gen_W(self):
        '''
        Creates mapping matrix from obs to action
        '''
        return np.random.randn(self.x,self.y)

    def gen_policy_list(self):
        '''Generate a pool or random policies'''
        return [self.gen_W() for _ in range(self.n_policy)]

    def policy_to_action(self,W, obs):
        return np.argmax(np.matmul(obs,W))

    def run_episode(self,policy=None, outdir = None):
        if policy is None:
            policy = self.best_policy
            self.env = wrappers.Monitor(self.env,directory=outdir,force=True)
        obs = self.env.reset()
        total_reward = 0
        while True:
            selected_action = self.policy_to_action(policy, obs)
            obs, reward, done, _ = self.env.step(selected_action)
            total_reward += reward
            if done:
                self.env.close()
                break
        return total_reward

    def train(self):
        policy_list = self.gen_policy_list()
        # Evaluate the score of each policy.
        scores_list = [self.run_episode(policy=p) for p in policy_list]
        # Select the best policy.
        print('Best policy score = {0:f}'.format(max(scores_list)))
        self.best_policy = policy_list[np.argmax(scores_list)]


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    outdir = 'Results'
    ## Ranom search among 5000 random policies
    agent  = RandomAgent(env,5000)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
