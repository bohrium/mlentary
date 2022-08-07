'''

We fix an 7 x 7 `world'.

 #  --- wall
h20 --- water
frn --- fern
chg --- robot charger
dsk --- desk
 @  --- robot

      0   1   2   3   4   5   6  
    .---.---.---.---.---.---.---.
  0 |h20|   |   |   |   |   |   | 
    :---:---:---:---:---:---:---:
  1 |   |   |   |   |   | # |   | 
    :---:---:---:---:---:---:---:
  2 |   |   | # | # | # | # |   | 
    :---:---:---:---:---:---:---:
  3 |   |   |   |   | # |chg|   | 
    :---:---:---:---:---:---:---:
  4 | # | # | # |[ ]| # |   | @ | 
    :---:---:---:---:---:---:---:
  5 |   |   |   |   | # |   |   | 
    :---:---:---:---:---:---:---:
  6 |   |frn|   |   |   |   |   | 
    '---'---'---'---'---'---'---'

Robot state:
{holding water, not holding water} x {charge 0, charge 33, charge 67, charge 100}

Reward Functions:
    -- water the fern
    -- water fern

Reward terms always in effect:
    -- make sure not run out of charge
    -- make sure not spill water anywhere but fern
    -- especially do not spill water on charger
    -- do not charge while holding water

'''

import tqdm

import numpy as np
np.random.seed(0)

class World:
    def __init__(self):
        self.__h20__ = (0,0)
        self.__frn__ = (6,1)
        self.__chg__ = (3,5)
        self.__dor__ = (4,3)
        self.__walls__ = {
                                                            #(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                                          (1,5),            #(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
                        (2,2),(2,3),(2,4),(2,5),            #(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                                    (3,4),                  #(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
            (4,0),(4,1),(4,2),      (4,4),                  #(4,0),(4,1),(4,2),(4,3),(4,4),(4,6),(4,6),
                                    (5,4),                  #(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),
                                                            #(6,0),(6,1),(6,2),(6,3),(6,4),(5,6),(6,6),
        }
        self.GAMMA = 0.98
        self.PROB_LOSE_CHARGE = 0.05
        self.PROB_FERN_DRY    = 0.015
        self.ACTIONS = {((dr,dc), w)  for dr in {-1,0,+1} 
                                      for dc in {-1,0,+1} if dr**2+dc**2<=1 
                                      for w in {True,False}}

        self.STATES = {((r,c),charge,hold, dry, door) for charge in range(4)
                                                      for hold in {True,False} 
                                                      for dry in {True,False} 
                                                      for door in {True}#{True,False}
                                                      for r in range(7)
                                                      for c in range(7) if self.in_bounds(r,c,door=door)
                                                      }
        self.reset()

    def state(self):
        return (self.position, self.charge, self.holding_water, self.fern_is_dry, self.door_is_open)

    def reset(self):
        #self.position = (5,6) # (row index, column index)
        #self.charge   = 3   # out of {0,1,2,3}
        #self.holding_water = True
        self.position, self.charge, self.holding_water, self.fern_is_dry, self.door_is_open = list(self.STATES)[np.random.choice(len(self.STATES))]
        self.steps_elapsed = 0 
        self.total_reward = 0

    def print_compact(self):
        print(self.position, self.charge, self.holding_water, self.number_of_steps_elapsed, self.total_reward)

    def print_verbose(self):
        print('       {:5d} steps so far; {:6.1f} total reward'.format(
            self.steps_elapsed, self.total_reward))
        print('state: {:3s} holding water; charge level {:3d}; fern is {:3s} dry'.format(
            '' if self.holding_water else 'NOT',
            round(33.33*self.charge),
            '' if self.fern_is_dry else 'NOT',
            ))
        print('.'+'---.'*7)
        for r in range(7):
            print('|', end='')
            for c in range(7):
                cell = '   '
                if (r,c) in self.__walls__: cell = ' # '
                if (r,c) == self.__frn__  : cell = 'frn' if self.fern_is_dry else 'FRN'
                if (r,c) == self.__chg__  : cell = 'chg'
                if (r,c) == self.__h20__  : cell = 'h20'
                if (r,c) == self.__dor__  : cell = '[ ]' if self.door_is_open else '[=]'
                #
                if (r,c) == self.position : cell = cell[0]+('@' if self.holding_water else 'O')+cell[2]  
                print(cell, end='')
                print('|', end='')
            print()
            print('.'+'---.'*7)

    def reset_cursor_after_print_verbose(self):
        print('\033[1A'*(2+(1+2*7)+2))

    def in_bounds(self, r,c, door=None):
        return (0<=r<7 and 0<=c<7 and (r,c) not in self.__walls__) and (
                (self.door_is_open or (r,c)!=self.__dor__) if door is None else ((r,c) != door)) 

    def perform_action(self, a):
        (dr, dc), water_action = a 
        reward = 0.0

        # door
        #if np.random.random()<0.01 :
        #    self.door_is_open = True
        #if np.random.random()<0.005:
        #    if self.position == self.__dor__:
        #        reward += -10.0  #prevented door from closing
        #    else:
        #        self.door_is_open = False #not self.door_is_open 

        # fern
        if np.random.random()<self.PROB_FERN_DRY:
            self.fern_is_dry = True

        if self.fern_is_dry: reward += - 1.0
        else:                reward += + 0.5

        #if self.holding_water:
        #    reward += + 0.1 # fun to hold water

        # water
        if water_action:
            reward += - 0.1 # cost for water action
            if self.position == self.__h20__:
                self.holding_water = not self.holding_water
            elif self.holding_water:
                self.holding_water = False
                if self.position == self.__frn__:
                    if not self.fern_is_dry:
                        reward += -10.0 # overwatered fern
                    self.fern_is_dry = False
                elif self.position == self.__chg__:
                    reward += -50.0 # watered charger
                else:
                    reward += - 1.0 # spilled water

        # locomotion
        if dr or dc:
            reward += - 0.05
        r, c = (self.position[0]+dr, self.position[1]+dc) 
        if self.in_bounds(r,c):
            self.position = (r,c)
        else:
            reward += - 0.1 # bumped into wall 

        # charging
        if self.position == self.__chg__:
            if self.charge==3:
                reward += - 0.5 # overcharged
            self.charge = min(3, self.charge+1) 
            if self.holding_water:
                reward += - 1.0 # charging while holding water
        elif np.random.random() < self.PROB_LOSE_CHARGE:
            self.charge = max(0, self.charge-1) 

        if self.charge == 0:
            reward += - 1.5     # cost of using backup battery

        self.steps_elapsed += 1 
        self.total_reward = reward + self.GAMMA * self.total_reward  
        return reward

    def simulate_policy(self, policy, nb_steps):
        self.reset()
        for _ in range(nb_steps):
            state = self.state()
            action = policy(state)
            self.perform_action(action)
        return self.total_reward

    def q_learn(self, policy, nb_lives, nb_steps_per_life, learning_rate=0.20, epsilon=0.1, optimism=5.0, curiosity=1.0, nb_pseudo_goals=20):
        q_table = {(s,a):optimism for s in self.STATES for a in self.ACTIONS}

        experience_buffer = []

        for l in tqdm.tqdm(range(nb_lives)):
            q_table_pseudo = {(s,a):q_table[(s,a)] for s in self.STATES for a in self.ACTIONS}
            #q_table_pseudo = {(s,a):1.0 for s in self.STATES for a in self.ACTIONS}

            pseudo_goals = set([
                    list(self.STATES)[np.random.choice(len(self.STATES))]
                    for _ in range(nb_pseudo_goals)#20)
                    ])

            self.reset()

            for t in (range(nb_steps_per_life)):
                state = self.state()
                action = policy(state) if np.random.random() < epsilon else (max((q_table_pseudo[(state,a)], a) for a in self.ACTIONS)[1])
                #action = policy(state) if np.random.random() < epsilon else (max((q_table[(state,a)], a) for a in self.ACTIONS)[1])
                r = self.perform_action(action)

                new_state = self.state()

                if np.random.random()<0.01:
                    experience_buffer.append((state, action, r, new_state))
                if len(experience_buffer) > 1000:
                    experience_buffer = experience_buffer[1:]

                q_table[(state,action)] += learning_rate * (
                    r + self.GAMMA * max(q_table[(new_state,a)] for a in self.ACTIONS)
                    - q_table[(state,action)]
                )

                if new_state in pseudo_goals:
                    r += curiosity

                q_table_pseudo[(state,action)] += learning_rate * (
                    r + self.GAMMA * max(q_table_pseudo[(new_state,a)] for a in self.ACTIONS)
                    - q_table_pseudo[(state,action)]
                )

                if experience_buffer:
                    for _ in range(2):
                        (state, action, r, new_state) = experience_buffer[np.random.choice(len(experience_buffer))]
                        q_table[(state,action)] += learning_rate * (
                            r + self.GAMMA * max(q_table[(new_state,a)] for a in self.ACTIONS)
                            - q_table[(state,action)]
                        )


        return q_table

W = World()
A = list(W.ACTIONS)
S = list(W.STATES)
print('{} many actions'.format(len(A)))
print('{} many states'.format(len(S)))

def uniform_policy(state):
    return A[np.random.choice(len(A))]

q_table = W.q_learn(uniform_policy,  1000, 1000)

def policy_from_table(state):
    return max((q_table[(state,a)], a) for a in W.ACTIONS)[1]

def flipbook(policy=uniform_policy):
    print()
    W.reset()
    W.print_verbose()
    W.reset_cursor_after_print_verbose()
    while True:
        input()
        a = policy(W.state())
        W.perform_action(a)
        W.print_verbose()
        print('just performed {}'.format(a), end='')
        W.reset_cursor_after_print_verbose()

flipbook(policy=policy_from_table)
