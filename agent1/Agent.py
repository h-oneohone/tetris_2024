import os
import numpy as np
import random


class Agent:
    def __init__(self, turn):
        dir_path = os.path.dirname(os.path.realpath(__file__))


    # def get_actions(self, state):
    #     actions = []
    #     for i in range(8):
    #         if state[i] == 0:
    #             actions.append(i)
    #     return actions

    # def choose_action(self, state):

    #     if len(self.current_actions) > 0:
    #         return self.current_actions.pop(0)
    #     else:
    #         self.current_actions = self.get_actions(state)
    #         return self.current_actions.pop(0)

    def choose_action(self, obs):
        return random.randint(0, 7)