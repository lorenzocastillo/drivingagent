import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from itertools import permutations,combinations
import sys
import pandas as pd
import numpy as np


MIN_INT = -sys.maxint - 1

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma = 0.8, alpha = 0.1, epsilon = 100):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = (None, None, None, None)
        self.Q = {self.state : [0,0,0,0]}
        self.decrement_epsilon = True
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        #STATS VARIABLES
        self.deadlines = []
        self.rewards = []
        self.reward_for_run = 0
        self.deadline_for_run = 0
    
#    def state_from_input(self,input,waypoint):
#        return (input['light'],input['left'],input['oncoming'],waypoint)
    #Eliminate redundant states
    def state_from_input(self,input,waypoint):
        if input['light'] == 'green':
            if waypoint == 'forward':
                return "Should Go Forward"
            elif waypoint == 'left':
                if input['oncoming'] == None:
                    return "Should Go Left"
                else:
                    return "Should Yield"
            else:
                return "Should Go Right"
        else:
            if waypoint == 'right':
                if input['left'] != 'forward':
                    return "Should Go Right"
                else:
                    return "Should Yield"
            else:
                return "Should Yield"
    def get_action_from_random(self):
        action_index = random.randrange(len(self.env.valid_actions))
        return action_index, self.env.valid_actions[action_index]

    #Pick the action with the best Q
    def get_action_from_policy(self,state):

        max_q = MIN_INT
        max_i = -1

        actions = []
        for i,q in enumerate(self.Q[state]):
            if q > max_q:
                actions.append((i,q))
                max_i = i
                max_q = q

        if max_i > -1:
            return (max_i, self.env.valid_actions[max_i])
        else:
            return self.get_action_from_random()

    #Picks an action based on either a random action or policy,
    #depending on an epsilon
    def get_action(self,state):
        random_number = random.randrange(100)
        if self.epsilon > random_number :
            return self.get_action_from_random()
        else:
            return self.get_action_from_policy(state)

    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Reduce the epsilon value to reduce randomness over time
        if self.decrement_epsilon:
            self.epsilon = max(5, self.epsilon - 2)

        #RESET STATS FOR RUN
        self.rewards.append(self.reward_for_run)
        self.deadlines.append(self.deadline_for_run)
        self.reward_for_run = 0
        self.deadline_for_run = 0
    


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.state_from_input(inputs, self.next_waypoint)
        
        
        if self.state not in self.Q:
            self.Q[self.state] = [0,0,0,0]
        
        # TODO: Select action according to your policy
        action_index , action = self.get_action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        q = reward + self.gamma * max(self.Q[self.state])
        self.Q[self.state][action_index] = (1 - self.alpha) * self.Q[self.state][action_index] + self.alpha * q
        
        #UPDATE STATS
        self.reward_for_run += reward
        self.deadline_for_run = deadline

#print self.Q


#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def test():
    results = open("Assignment 4 Results",'w')
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



#        print "Rewards: ", a.rewards
#        frame = pd.DataFrame(data = a.rewards[len(a.rewards)/2:])
#        print frame.describe()
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
