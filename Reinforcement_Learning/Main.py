
from Agents import *
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.action_space.sample()

    outdir = 'Results'

    ## Ranom search among 5000 random policies
    #agent  = RandomAgent(env,5000)

    ## Genetic evolution with 20 policies and 100 steps.
    ## It does not guarantee to converge
    #agent = GeneticAgent(env,20,100)

    ## Q learning. Sovse iterratively Bellman equation for a Q-function
    agent = QAgent(env,10000)


    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
