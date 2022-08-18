''' author: samtenka
    change: 2022-08-06
    create: 2022-08-07
    descrp:
    to use:

We fix an 7 x 7 `world'.

 #  --- wall
h20 --- water
frn --- fern
chg --- robot charger
dsk --- desk
bkt --- bucket
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
  6 |   |frn|   |   |   |   |bkt| 
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

ENABLE_CHARGE = True#False
ENABLE_DOOR   = False#False
ENABLE_BUCKET = False#False

GRAY   = '\033[30m' 
RED    = '\033[31m' 
GREEN  = '\033[32m' 
YELLOW = '\033[33m' 
BLUE   = '\033[34m' 
MAGENTA= '\033[35m' 
CYAN   = '\033[36m' 

class QFunction:
    def __init__(self, STATES, ACTIONS, optimism=0.0):
        # initialize all q table values to 0
        self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}
        self.ACTIONS = ACTIONS

    def copy_from(self, rhs):
        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
        self.ACTIONS = rhs.ACTIONS

    def update(self, s, a, new_val, learning_rate):
        # INTERESTING LINE: 
        # update self.q_table[(s,a)] to be closer to new_val
        self.q_table[(s,a)] += learning_rate * (new_val - self.q_table[(s,a)])

    def query_q(self, s, a):
        # gets a value from the q table
        return self.q_table[(s,a)]

    def query_a(self, s):
        # s is an input
        # a --- some action --- is the output
        #
        # In particular, this reads off an OPTIMAL POLICY given the current
        # q-table values 
        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
        return a

    def query_aq(self, s):
        # this gives us the VALUE FUNCTION of s given the current q values 
        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
        return q

#class QFunction:
#    def __init__(self, STATES, ACTIONS, optimism=0.0):
#        self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}
#
#        example_state = next(iter(STATES))
#        self.state_axes = [set([]) for _ in example_state] 
#        for s in STATES:
#            for i,x in enumerate(s):
#                self.state_axes[i].add(x)
#        print(self.state_axes)
#        self.q_aux = [{(x,a):0.0 for x in axis for a in ACTIONS} for axis in self.state_axes]
#        self.ACTIONS = ACTIONS
#    def copy_from(self, rhs):
#        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
#        self.q_aux   = [{k:d[k] for k in d} for d in rhs.q_aux]
#        self.ACTIONS = rhs.ACTIONS
#    def update(self, s, a, new_val, learning_rate, aux_factor=0.000):
#        discrepancy = new_val - self.query_q(s,a) 
#        self.q_table[(s,a)] += learning_rate * discrepancy
#        for i,x in enumerate(s):
#            self.q_aux[i][(x,a)] += learning_rate*(aux_factor * discrepancy)
#    def query_q(self, s, a):
#        return self.q_table[(s,a)] + sum(self.q_aux[i][(x,a)] for i,x in enumerate(s))
#    def query_a(self, s):
#        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
#        return a
#    def query_aq(self, s):
#        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
#        return q

class World:
    def __init__(self):
        self.__h20__ = (0,0)
        self.__frn__ = (6,1)
        self.__bkt__ = (6,6)
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
        self.PROB_DOOR_OPENS  = 0.015
        self.PROB_DOOR_CLOSES = 0.010
        self.PROB_LOSE_CHARGE = 0.020
        self.PROB_FERN_DRY    = 0.025
        self.ACTIONS = {((dr,dc), w)  for dr in {-1,0,+1} 
                                      for dc in {-1,0,+1} if dr**2+dc**2<=1 
                                      for w in {True,False}}

        self.STATES = {( r,c ,charge,hold, dry, door, bucket)
                       for charge in (range(4) if ENABLE_CHARGE else [3])
                       for hold in {True,False} 
                       for dry in {True,False} 
                       for door in ({True,False} if ENABLE_DOOR else {True})
                       for bucket in ({True,False} if ENABLE_BUCKET else {True})
                       for r in range(7)
                       for c in range(7) if self.in_bounds(r,c,door=door)
                       }
        self.reset()

    def state(self):
        return (self.row, self.col, self.charge, self.holding_water, self.fern_is_dry, self.door_is_open, self.bucket_is_empty)

    def reset(self):
        self.row, self.col, self.charge, self.holding_water, self.fern_is_dry, self.door_is_open, self.bucket_is_empty = list(self.STATES)[np.random.choice(len(self.STATES))]
        self.steps_elapsed = 0 
        self.total_reward = 0

    #def print_compact(self):
    #    print(self.position, self.charge, self.holding_water, self.number_of_steps_elapsed, self.total_reward)

    def print_verbose(self):
        print(YELLOW,end='')
        print('       {:5d} steps so far; {:6.1f} total reward'.format(
            self.steps_elapsed, self.total_reward))
        print('state: {}{:13s}{}; charge level {}{:3d}{}; fern is {}{:3s}{}'.format(
            BLUE,'holding water' if self.holding_water else '',YELLOW,
            BLUE,round(33.33*self.charge),YELLOW,
            BLUE,'DRY' if self.fern_is_dry else 'ok',YELLOW,
            ))
        print(GRAY,end='')
        print(BLUE+'.'+     '---.'*6+'---'+     '.'+GRAY)
        for r in range(7):
            print(BLUE+'|'+GRAY, end='')
            for c in range(7):
                cell = '   '
                # TODO: fix coloration of supimposed robot on cell!!
                if (r,c) in self.__walls__: cell = BLUE+' # '+GRAY
                if (r,c) == self.__frn__  : cell = (YELLOW+'frn'+GRAY) if self.fern_is_dry else (GREEN+'FRN'+GRAY)
                if (r,c) == self.__chg__  : cell = YELLOW+'chg'+GRAY
                if (r,c) == self.__bkt__  : cell = (YELLOW+'bkt'+GRAY) if self.bucket_is_empty else (CYAN+'BKT'+GRAY)
                if (r,c) == self.__h20__  : cell = CYAN+'h20'+GRAY
                if (r,c) == self.__dor__  : cell = '[ ]' if self.door_is_open else (MAGENTA+'[=]'+GRAY)
                #
                if (r,c) == (self.row, self.col): cell = cell[0]+((CYAN+'@') if self.holding_water else (RED+'O'))+GRAY+cell[2]  
                print(cell, end='')
                if c+1 != 7:
                    print('|', end='')
                else:
                    print(BLUE+'|'+GRAY, end='')
            print()
            if r+1 != 7:
                print(BLUE+':'+GRAY+'---:'*6+'---'+BLUE+':'+GRAY)
            else:
                print(BLUE+"'"+     "---'"*6+'---'+     "'"+GRAY)

    def reset_cursor_after_print_verbose(self):
        print('\033[1A'*(2+(1+2*7)+2))

    def in_bounds(self, r,c, door=None):
        return (0<=r<7 and 0<=c<7 and (r,c) not in self.__walls__) and (
                (self.door_is_open or (r,c)!=self.__dor__) if door is None else ((r,c) != door)) 

    def perform_action(self, a):
        (dr, dc), water_action = a 
        reward = 0.0

        pos = self.row, self.col

        # door
        if ENABLE_DOOR:
            if np.random.random() < self.PROB_DOOR_OPENS:
                self.door_is_open = True
            if np.random.random() < self.PROB_DOOR_CLOSES:
                if (self.row,self.col) == self.__dor__:
                    reward += -10.0  #prevented door from closing
                else:
                    self.door_is_open = False

        # fern
        if np.random.random()<self.PROB_FERN_DRY:
            self.fern_is_dry = True
        #
        if self.fern_is_dry: reward += - 1.5
        else:                reward += + 0.2 ############ POSITIVE REWARD

        #if self.holding_water:
        #    reward += + 0.1 # fun to hold water############ POSITIVE REWARD

        # water
        if (self.row,self.col) == self.__h20__ and not self.fern_is_dry:
            reward += + 2.0
        if water_action:
            reward += - 0.1 # cost for water action
            if (self.row,self.col) == self.__h20__:
                self.holding_water = not self.holding_water
            elif ENABLE_BUCKET and (self.row,self.col) == self.__bkt__:
                self.holding_water, self.bucket_is_empty = not self.bucket_is_empty, not self.holding_water # swap contents
            elif self.holding_water:
                self.holding_water = False
                if (self.row,self.col) == self.__frn__:
                    if self.fern_is_dry:
                        reward += + 1.0 # hooray!             ############ POSITIVE REWARD
                    if not self.fern_is_dry:
                        reward += -10.0 # overwatered fern
                    self.fern_is_dry = False
                elif (self.row,self.col) == self.__chg__:
                    reward += -20.0 # watered charger
                else:
                    reward += - 2.0 # spilled water

        # charging
        if ENABLE_CHARGE:
            if (self.row,self.col) == self.__chg__:
                if self.charge==3:
                    reward += - 0.5 # overcharged
                self.charge = min(3, self.charge+1) 
                if self.holding_water:
                    reward += - 2.0 # charging while holding water
            #elif (r or c) and np.random.random() < self.PROB_LOSE_CHARGE:
            #    self.charge = max(0, self.charge-1) # indirect cost of motion 
            #
            if self.charge == 0:
                reward += - 1.5     # cost of using backup battery

        # locomotion (should be last paragraph due to pos variable)
        if dr or dc:
            reward += - 0.1 # direct cost of motion (annoying sound)
        r, c = (self.row+dr, self.col+dc) 
        if self.in_bounds(r,c):
            self.row, self.col = (r,c)
        else:
            reward += - 0.5 # bumped into wall 

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

    # 4 methods (okay to mix) to encourage exploration: 
    #   ``EPSILON GREEDY''
    #   ``OPTIMISM IN THE FACE OF UNCERTAINTY''
    #   ``PSEUDO-GOALS''
    #   #``EXPERIENCE REPLAY''

    def q_learn(self, policy, nb_lives, nb_steps_per_life,
                learning_rate = 0.1 , epsilon = 0.1,
                nb_buffer_updates = 2, buffer_prob = 0.01, buffer_size = 1000,
                optimism = +10.0,
                pseudo_goal_bonus = 1.0, nb_pseudo_goals = 20):

        q_main    = QFunction(self.STATES, self.ACTIONS, optimism)
        q_explore = QFunction(self.STATES, self.ACTIONS)

        experience_buffer = []

        for l in tqdm.tqdm(range(nb_lives)):
            eps = epsilon / (l+1)
            #eps = np.random.choice([0.01,0.1,0.5,0.9,0.99])
            q_explore.copy_from(q_main)

            pseudo_goals = set([
                    list(self.STATES)[np.random.choice(len(self.STATES))]
                    for _ in range(nb_pseudo_goals)
                    ])

            self.reset()

            for t in (range(nb_steps_per_life)):
                s = self.state()
                a = policy(s) if np.random.random() < eps else q_explore.query_a(s)
                r = self.perform_action(a)
                new_state = self.state()

                # THIS PARAGAPH IS THE ESSENCE OF Q LEARNING
                s_prime = new_state
                noisy_estimate_of_true_Q = r + self.GAMMA * max([
                    q_main.q_table[(s_prime,a_prime)]
                    for a_prime in self.ACTIONS])   
                new_val = noisy_estimate_of_true_Q 
                q_main.update(s,a,new_val, learning_rate)

                if new_state in pseudo_goals:
                    r += pseudo_goal_bonus 
                noisy_estimate_of_true_Q = r + self.GAMMA * max([
                    q_explore.q_table[(s_prime,a_prime)]
                    for a_prime in self.ACTIONS])   
                q_explore.update(s,a,new_val, learning_rate)

        return q_main

print(YELLOW,end='')
W = World()
A = list(W.ACTIONS)
S = list(W.STATES)
print('{} many actions'.format(len(A)))
print('{} many states'.format(len(S)))

def uniform_policy(state):
    return A[np.random.choice(len(A))]
q_table = W.q_learn(uniform_policy,  3000, 1500)
np.save('testing.npy', q_table)
#q_table = np.load('q_table-1000-1000.npy', allow_pickle=True).item()
#q_table = np.load('charge0-door0-bucket0-lin-q_table-500-1000.npy', allow_pickle=True).item()
q_table = np.load('testing.npy', allow_pickle=True).item()

def policy_from_table(state):
    return q_table.query_a(state)

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
        print(YELLOW+'just performed {}     '+GRAY.format(a), end='')
        W.reset_cursor_after_print_verbose()

np.random.random()
flipbook(policy=policy_from_table)
#flipbook(policy=uniform_policy)

