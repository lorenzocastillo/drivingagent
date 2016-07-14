import random
import sys
import pandas as pd
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from itertools import permutations,combinations


MIN_INT = -(sys.maxint - 1)
SHOULD_FORWARD = 0
SHOULD_LEFT = 1
SHOULD_RIGHT = 2
SHOULD_YIELD = 3

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma=0.8, alpha=0.1, epsilon=100):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)

        self.state = None
        self.Q = {self.state: [0, 0, 0, 0]}
        self.decrement_epsilon = True
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        #STATS VARIABLES
        self.deadlines = list()
        self.rewards = list()
        self.reward_for_run = 0
        self.deadline_for_run = 0

    def state_from_input(self, input, waypoint):
        """
        Reduces the Action Space based on states
        :param input:
        :param waypoint:
        :return:
        """
        if input['light'] == 'green':
            if waypoint == 'forward':
                return SHOULD_FORWARD
            elif waypoint == 'left':
                if input['oncoming'] is None:
                    return SHOULD_LEFT
                else:
                    return SHOULD_YIELD
            else:
                return SHOULD_RIGHT
        else:
            if waypoint == 'right':
                if input['left'] != 'forward':
                    return SHOULD_RIGHT
                else:
                    return SHOULD_YIELD
            else:
                return SHOULD_YIELD

    def get_action_from_random(self):
        """
        Make a random action
        :return:
        """
        action_index = random.randrange(len(self.env.valid_actions))
        return action_index, self.env.valid_actions[action_index]

    def get_action_from_policy(self, state):
        """
        Pick the action with the best Q score
        :param state:
        :return:
        """
        max_i, max_q = max(enumerate(self.Q[state]), key=lambda x: x[1])

        if max_i > -1:
            return max_i, self.env.valid_actions[max_i]
        else:
            return self.get_action_from_random()

    def get_action(self,state):
        """
        Picks an action based on either a random action or policy, depending on an epsilon
        :param state:
        :return:
        """
        random_number = random.randrange(100)
        if self.epsilon > random_number :
            return self.get_action_from_random()
        else:
            return self.get_action_from_policy(state)

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # Reduce the epsilon value to reduce randomness over time
        if self.decrement_epsilon:
            self.epsilon = max(5, self.epsilon - 2)

        #  RESET STATS FOR RUN
        self.rewards.append(self.reward_for_run)
        self.deadlines.append(self.deadline_for_run)
        self.reward_for_run = 0
        self.deadline_for_run = 0

    def update(self, t):
        """
        Makes an action and updates Q-Score
        :param t:
        :return:
        """
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = self.state_from_input(inputs, self.next_waypoint)
        
        if self.state not in self.Q:
            self.Q[self.state] = [0, 0, 0, 0]

        action_index , action = self.get_action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        q = reward + self.gamma * max(self.Q[self.state])
        self.Q[self.state][action_index] = (1 - self.alpha) * self.Q[self.state][action_index] + self.alpha * q

        self.reward_for_run += reward
        self.deadline_for_run = deadline


def test():
    results = open("Results",'w')
    results.write("STATS\n")
    
    """Run the agent for a finite number of trials."""
    gammas = [0.05, 0.1, 0.3, 0.5]
    alphas = [0.9, 0.8, 0.5]
    
    combs = [(g,a) for g in gammas for a in alphas]
        
    for (gamma,alpha) in combs:
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        a.gamma = gamma
        a.alpha = alpha
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

        # Now simulate it
        sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
        sim.run(n_trials=75)  # press Esc or close pygame window to quit

        results.write("Parameters: \n")
        results.write("Alpha: ")
        results.write(str(a.alpha))
        results.write(" Gamma: ")
        results.write(str(a.gamma))
        results.write("\nDeadlines: \n")
        results.write(str(a.deadlines))
        results.write("\n")
        deadline_frame = pd.DataFrame(data = a.deadlines[len(a.deadlines)/2:])
        results.write(str(deadline_frame.describe()))

        results.write("\nQ Table: \n")
        results.write(str(a.Q))
        results.write("\n\n\n")
    results.close()


def run():
    #test()
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    print "STATS"
    print "Rewards: ", a.rewards
    frame = pd.DataFrame(data = a.rewards[50:])
    print frame.describe()
    
    print "Deadlines: ", a.deadlines
    deadline_frame = pd.DataFrame(data = a.deadlines[50:])
    print deadline_frame.describe()
    
    print "Q Table: "
    print a.Q

if __name__ == '__main__':
    run()
