import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys
import pandas as pd
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_states = ["Start","Should Go Forward", "Should Go Left", "Should Go Right", "Breaking Law"]
        self.state = self.valid_states[0]
        self.Q = {}
        self.epsilon = 100
        self.gamma = 0.9
        self.alpha = 0.1
        
        #Stats
        self.deadlines = []
        self.rewards = []
        self.reward_for_run = 0
        self.deadline_for_run = 0
    
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

    #Choose the option that never gets penalized
    #For test purposes only
    def get_optimal_action(self, state):
        if state == "Should Go Right":
            return (3,'right')
        elif state == "Should Go Left":
            return (2, 'left')
        elif state == "Should Go Forward":
            return (1, 'forward')
        elif state == "Should Yield":
            return (0, None)
        else:
            return self.get_action_from_random()

    #Pick the action with the best Q
    def get_action_from_policy(self,state):

        max_q = -1
        max_i = -1

        actions = []
        for i,q in enumerate(self.Q[state]):
            if q > max_q:
                actions.append((i,q))
                max_i = i
                max_q = q


        return (max_i, self.env.valid_actions[max_i])

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
        self.epsilon = max(0, self.epsilon - 5)

        self.rewards.append(self.reward_for_run)
        self.deadlines.append(self.deadline_for_run)

        #RESET STATS FOR RUN
        self.reward_for_run = 0
        self.deadline_for_run = 0
    


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        next_state = self.state_from_input(inputs, self.next_waypoint)
        
        if next_state not in self.Q:
            self.Q[next_state] = [0,0,0,0]
        
        # TODO: Select action according to your policy
        action_index , action = self.get_action(next_state)

        # Execute action and get reward
        reward = self.env.act(self, action)


        q = reward + self.gamma * self.Q[next_state][action_index]
        self.Q[next_state][action_index] = (1 - self.alpha) * q + self.alpha * self.Q[next_state][action_index]
        self.state = next_state
        
        #UPDATE STATS
        self.reward_for_run += reward
        self.deadline_for_run = deadline


#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
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
