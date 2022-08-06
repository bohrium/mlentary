'''

We fix an 8 x 8 `world'.

 #  --- wall
h20 --- water
frn --- fern
chg --- robot charger
dsk --- desk
 @  --- robot

      0   1   2   3   4   5   6   7  
    .---.---.---.---.---.---.---.---.
  0 |h20|   |   |   |   |   |   |   | 
    :---:---:---:---:---:---:---:---:
  1 |   |   |   |   |   | # |   |   | 
    :---:---:---:---:---:---:---:---:
  2 |   |   | # | # | # | # |   |   | 
    :---:---:---:---:---:---:---:---:
  3 |   |   |   |   | # |chg|   |   | 
    :---:---:---:---:---:---:---:---:
  4 |   |   |   |   | # |   |   |   | 
    :---:---:---:---:---:---:---:---:
  5 | # | # | # |   | # |   | @ |   | 
    :---:---:---:---:---:---:---:---:
  6 |   |   |   |   | # |   |   |   | 
    :---:---:---:---:---:---:---:---:
  7 |   |frn|   |   |   |   |   |   | 
    '---'---'---'---'---'---'---'---'

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

import numpy as np
np.random.seed(0)

class World:
    def __init__(self):
        self.__h20__ = (0,0)
        self.__frn__ = (7,1)
        self.__chg__ = (3,5)
        self.__walls__ = {
                                                            #(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
                                          (1,5),            #(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
                        (2,2),(2,3),(2,4),(2,5),            #(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),
                                    (3,4),                  #(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),
                                    (4,4),                  #(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),
            (5,0),(5,1),(5,2),      (5,4),                  #(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),
                                    (6,4),                  #(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),
                                                            #(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),
        }
        self.reset()
        self.GAMMA = 0.98
        self.PROB_LOSE_CHARGE = 0.05
        self.ACTIONS = {((r,c), w)  for r in {-1,0,+1} 
                                    for c in {-1,0,+1} if r**2+c**2<=1 
                                    for w in {True,False}}

    def reset(self):
        self.position = (5,6) # (row index, column index)
        self.charge   = 3   # out of {0,1,2,3}
        self.holding_water = True
        self.steps_elapsed = 0 
        self.total_reward = 0

    def print_compact(self):
        print(self.position, self.charge, self.holding_water, self.number_of_steps_elapsed, self.total_reward)

    def print_verbose(self):
        print('state: {:3s} holding water; charge level {:3d}'.format(
            '' if self.holding_water else 'NOT',
            round(33.33*self.charge)))
        print('       {:5d} steps so far; {:6.1f} total reward'.format(
            self.steps_elapsed, self.total_reward))
        print('.'+'---.'*8)
        for r in range(8):
            print('|', end='')
            for c in range(8):
                cell = '   '
                if (r,c) in self.__walls__: cell = ' # '
                if (r,c) == self.__frn__  : cell = 'frn'
                if (r,c) == self.__chg__  : cell = 'chg'
                if (r,c) == self.__h20__  : cell = 'h20'
                #
                if (r,c) == self.position : cell = cell[0]+'@'+cell[2]
                print(cell, end='')
                print('|', end='')
            print()
            print('.'+'---.'*8)

    def reset_cursor_after_print_verbose(self):
        print('\033[1A'*(2+(1+2*8)+2))

    def in_bounds(self, r,c):
        return 0<=r<8 and 0<=c<8 and (r,c) not in self.__walls__

    def perform_action(self, a):
        (dr, dc), water_action = a 
        reward = 0.0

        # water
        if water_action:
            reward += - 0.1 # cost for water action
            if self.position == self.__h20__:
                self.holding_water = not self.holding_water
            elif self.holding_water:
                self.holding_water = False
                if self.position == self.__frn__:
                    reward += + 1.0 # watered fern
                elif self.position == self.__chg__:
                    reward += -10.0 # watered charger
                else:
                    reward += - 2.0 # spilled water

        # locomotion
        r, c = (self.position[0]+dr, self.position[1]+dc) 
        if self.in_bounds(r,c):
            self.position = (r,c)
        else:
            reward += - 0.1 # bumped into wall 

        # charging
        if self.position == self.__chg__:
            self.charge = min(3, self.charge+1) 
            if self.holding_water:
                reward += - 5.0 # charging while holding water
        elif np.random.random() < self.PROB_LOSE_CHARGE:
            self.charge = max(0, self.charge-1) 

        if self.charge == 0:
            reward += - 0.5 # cost of using backup battery

        self.steps_elapsed += 1 
        self.total_reward = reward + self.GAMMA * self.total_reward  
        return reward

W = World()
A = list(W.ACTIONS)
print(A)
print()
print()
print()
input()

W.print_verbose()
W.reset_cursor_after_print_verbose()
while True:
    #input()
    a = A[np.random.choice(len(A))]
    #print(a)
    W.perform_action(a)
    W.print_verbose()
    print('just performed {}'.format(a), end='')
    W.reset_cursor_after_print_verbose()
