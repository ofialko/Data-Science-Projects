import gym
from gym import wrappers
import numpy as np

class GeneticAgent(object):
    '''Genetic agent: maintains a pool of candidate solutions.
    Iteratively, the generation evolved to produce the next generation
    which has candidate solutions with higher fitness values than the previous generation.
    This process is repeated for a pre-specified number of generations or until
    a solution with goal fitness value is found.'''
    def __init__(self,env,n_policy,n_steps):
        self.env = env
        self.n_policy = n_policy
        self.n_steps = n_steps

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

    def crossover(self,policy1, policy2):
        new_policy = policy1.copy()
        for i in range(self.x):
            for j in range(self.y):
                rand = np.random.uniform()
                if rand > 0.5:
                    new_policy[i,j] = policy2[i,j]
        return new_policy

    def mutation(self,policy, p=0.05):
        new_policy = policy.copy()
        for i in range(self.x):
            for j in range(self.y):
                rand = np.random.uniform()
                if rand < p:
                    new_policy[i,j] = np.random.randn()
        return new_policy


    def train(self):
        policy_list = self.gen_policy_list()
        for idx in range(self.n_steps):
            # Evaluate the score of each policy
            policy_scores = [self.run_episode(policy=p) for p in policy_list]
            print('Generation {0:d} : max score = {1:0.2f}'.format(idx+1, max(policy_scores)))
            policy_ranks = list(reversed(np.argsort(policy_scores)))
            elite_set = [policy_list[x] for x in policy_ranks[:5]]
            select_probs = np.array(policy_scores) / np.sum(policy_scores)
            child_set = [self.crossover(
                policy_list[np.random.choice(range(self.n_policy), p=select_probs)],
                policy_list[np.random.choice(range(self.n_policy), p=select_probs)])
                for _ in range(self.n_policy - 5)]

            mutated_list = [self.mutation(p) for p in child_set]
            policy_list = elite_set
            policy_list += mutated_list

        policy_scores = [self.run_episode(policy=p) for p in policy_list]
        print('Best policy score = {0:f}'.format(max(policy_scores)))
        self.best_policy = policy_list[np.argmax(policy_scores)]

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    outdir = 'Results'

    ## Genetic evolution with 20 policies and 100 steps.
    ## It does not guarantee to converge
    agent = GeneticAgent(env,20,100)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
