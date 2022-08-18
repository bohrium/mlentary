''' author: samtenka
    change: 2022-08-17
    create: 2022-08-07
    descrp:  
            See section 0 below for details.
    jargon:
    to use:
'''

#==============================================================================
#===  0. OVERVIEW OF MARKOV DECISION PROCESS  =================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.0. States of the World  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    We aim to train a robot to act in a fixed 7 x 7 house whose floorplan looks
    like this:
    
          0   1   2   3   4   5   6             
        .---.---.---.---.---.---.---.           LEGEND:
      0 |h20|   |   |   |   |   |   |                                  
        :---:---:---:---:---:---:---:            #  --- wall          
      1 |   |   |   |   |   | # |   |           h20 --- water source
        :---:---:---:---:---:---:---:           frn --- fern
      2 |   |   | # | # | # | # |   |            @  --- ROBOT           
        :---:---:---:---:---:---:---:                         
      3 |   |   |   |   | # |chg|   |           bkt --- bucket                  
        :---:---:---:---:---:---:---:           chg --- robot charger                     
      4 | # | # | # |[ ]| # |   | @ |           [ ] --- door 
        :---:---:---:---:---:---:---:           
      5 |   |   |   |   | # |   |   |           
        :---:---:---:---:---:---:---:
      6 |   |frn|   |   |   |   |bkt| 
        '---'---'---'---'---'---'---'
    
    The robot and no other elements of the world may move around.  The state of
    the world is specified by the robot's location together the internal states
    of the robot, fern, bucket, door.
    
    The robot's internal state has 6 values:
        {has water, not has water} x
        {charge 0, charge 33, charge 67, charge 100}
    
    The fern's internal state has 2 values:
        {dry, wet}
    
    The bucket's internal state has 2 values:
        {empty, full of water}
    
    The door's internal state has 2 values:
        {open, closed}

    There are thus roughly 7*7 * 6 * 2 * 2 * 2 states in total.  Actually,
    since the robot may never sit atop a wall or a closed door, there are
    slightly fewer states than that.

    At each timestep, the robot knows the exact state of the world --- the
    human residents have equipped the door, bucket, and fern with internet
    connected sensors.
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.1. Actions and Dynamics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    At each timestep, the robot may stay put or attempt to move
    up/down/left/right (successful unless it tries to move into a wall, into a
    closed door, or off the grid).

    The robot may also perform a `water action' to pick up water (successful if
    it is at h20 or at a full-of-water bucket) or to expel water (successful
    anywhere).  If it expels water onto a dry fern, the fern becomes wet; if it
    expels water onto a bucket, the bucket becomes full. 

    The robot has no means to open or close the door.  The door occasionally
    opens or closes by itself (this models the effect of unseen humans living
    in the house).

    The robot gradually loses charge over time.  It increases its charge by
    sitting atop the charging station.
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.2. Rewards  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    The robot experiences a reward that is a sum of several terms: 
        {a} [medium]    like when the fern is wet
        {b} [medium] dislike when the fern is dry

        {c} [STRONG] dislike of watering an already wet fern
        {d} [medium]    like of watering a dry fern
        {e} [STRONG] dislike of spilling water on charger
        {f} [medium] dislike of spilling water anywhere but fern or bucket

        {g} [ mild ] dislike of overcharging once full of charge
        {h} [medium] dislike of charging while has water
        {i} [STRONG] dislike when no charge left (backup battery expensive)

        {j} [STRONG] dislike of blocking door from being closed

        {k} [ mild ] dislike of moving at all (motors make annoying sound)
        {l} [medium] dislike of bumping into the wall or into a closed door

    The discount factor is GAMMA=0.98, meaning that the robot cares most about
    its rewards within the next 50=1/(1-GAMMA) timesteps.  
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.3. Simplifications  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    By default, we work in a simpler world wherein:
    -- the door is ALWAYS open and never closes
    -- the bucket is ALWAYS empty and can never be filled
    -- the robot's charge is ALWAYS 100% and the charging station has no effect

    In this simpler world, the robot just needs to learn to ferry water from
    the h20 source to the fern whenever the fern becomes dry.

    We may relax these constraints and allow the door, bucket, and/or charge
    to change over time by flipping False->True in these lines of code:
        ENABLE_CHARGE = False
        ENABLE_DOOR   = False
        ENABLE_BUCKET = False
'''

#==============================================================================
#===  1. GLOBAL CONSTANTS  ====================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  1.0. Imported Modules  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tqdm
import time
import numpy as np

def coin(p):
    return np.random.random() < p
def random_el(L):
    return L[np.random.choice(len(L))]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  1.1. Simulation Parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ENABLE_CHARGE = True        #   True or False
ENABLE_DOOR   = False       #   True or False
ENABLE_BUCKET = False       #   True or False

Q_FUNCTION = 'tabular'      #   'tabular' or 'linear'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  1.2. Color Printing  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRAY   = '\033[30m' 
RED    = '\033[31m' 
GREEN  = '\033[32m' 
YELLOW = '\033[33m' 
BLUE   = '\033[34m' 
MAGENTA= '\033[35m' 
CYAN   = '\033[36m' 

COLORS_BY_TAGS = {
  'a': GRAY   , 
  'r': RED    , 
  'e': GREEN  , 
  'y': YELLOW , 
  'b': BLUE   , 
  'm': MAGENTA, 
  'c': CYAN   , 
}

def display(s, *args):
    s = s.format(*args)
    for k,v in COLORS_BY_TAGS.items():
        s = s.replace('<'+k+'>', v)
    print(s, end='')

#==============================================================================
#===  2. Q-FUNCTIONS AS DATA STRUCTURES  ======================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  2.0. Tabular Q-Function  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QFunction:#Tabular:
    def __init__(self, STATES, ACTIONS, optimism=0.0):
        self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}
        self.ACTIONS = ACTIONS
    def copy_from(self, rhs):
        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
        self.ACTIONS = rhs.ACTIONS
    def update(self, s, a, new_val, learn_rate):
        self.q_table[(s,a)] += learn_rate * (new_val - self.q_table[(s,a)])
    def query_q(self, s, a):
        return self.q_table[(s,a)]
    def query_a(self, s):
        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
        return a
    def query_aq(self, s):
        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
        return q

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  2.1. Linear Q-Function  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QFunctionLinear:
    def __init__(self, STATES, ACTIONS, optimism=0.0):
        self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}

        example_state = next(iter(STATES))
        self.state_axes = [set([]) for _ in example_state] 
        for s in STATES:
            for i,x in enumerate(s):
                self.state_axes[i].add(x)
        self.q_aux = [{(x,a):0.0 for x in axis for a in ACTIONS}
                      for axis in self.state_axes               ]
        self.ACTIONS = ACTIONS
    def copy_from(self, rhs):
        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
        self.q_aux   = [{k:d[k] for k in d} for d in rhs.q_aux]
        self.ACTIONS = rhs.ACTIONS
    def update(self, s, a, new_val, learn_rate, aux_factor=0.000):
        discrepancy = new_val - self.query_q(s,a) 
        self.q_table[(s,a)] += learn_rate * discrepancy
        for i,x in enumerate(s):
            self.q_aux[i][(x,a)] += learn_rate*(aux_factor * discrepancy)
    def query_q(self, s, a):
        return self.q_table[(s,a)] + sum(self.q_aux[i][(x,a)]
                                         for i,x in enumerate(s))
    def query_a(self, s):
        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
        return a
    def query_aq(self, s):
        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
        return q

#==============================================================================
#===  3. IMPLEMENTING THE MDP  ================================================
#==============================================================================

class World:
    def __init__(self):
        self.__h20__ = (0,0)
        self.__frn__ = (6,1)
        self.__bkt__ = (6,6)
        self.__chg__ = (3,5)
        self.__dor__ = (4,3)
        self.__walls__ = {
                                                            #
                                          (1,5),            #
                        (2,2),(2,3),(2,4),(2,5),            #
                                    (3,4),                  #
            (4,0),(4,1),(4,2),      (4,4),                  #
                                    (5,4),                  #
                                                            #
        }
        self.GAMMA = 0.98
        self.PROB_DOOR_OPENS  = 0.020
        self.PROB_DOOR_CLOSES = 0.015
        self.PROB_LOSE_CHARGE = 0.025
        self.PROB_FERN_DRY    = 0.030
        self.ACTIONS = {((dr,dc), w)  for dr in {-1,0,+1} 
                                      for dc in {-1,0,+1} if dr**2+dc**2<=1 
                                      for w in {True,False}}

        self.STATES = [(r,c,has_water,charge,fern_wet,bucket_full,door_open)
                       for has_water in {True,False} 
                       for charge in (range(4) if ENABLE_CHARGE else [3])
                       for fern_wet in {True,False} 
                       for bucket_full in ({True,False} if ENABLE_BUCKET else {False})
                       for door_open in ({True,False} if ENABLE_DOOR else {True})
                       for r in range(7)
                       for c in range(7) if self.in_bounds(r,c,door=door_open)
                      ]
        self.reset()

    def state(self):
        return (self.row, self.col, self.has_water, self.charge,
                self.fern_wet, self.bucket_full, self.door_open)

    def perform_action(self, a):
        self.environment_just_changed = False

        (dr, dc), water_action = a 
        reward = 0.0

        # fern
        if self.fern_wet and coin(self.PROB_FERN_DRY):
            self.fern_wet = False
            self.environment_just_changed = True
        if self.fern_wet: reward += + 0.5 # reward term named {a}
        else:             reward += - 0.5 # reward term named {b}

        # water
        if water_action:
            reward += - 0.1 # cost for water action
            if (self.row,self.col) == self.__h20__:
                self.has_water = not self.has_water
            elif ENABLE_BUCKET and (self.row,self.col) == self.__bkt__:
                # swap contents:
                self.has_water, self.bucket_full = self.bucket_full, self.has_water
                self.environment_just_changed = True
            elif self.has_water:
                self.has_water = False
                if (self.row,self.col) == self.__frn__:
                    if self.fern_wet:
                        reward += -10.0 # reward term named {c}
                    else:
                        reward += + 1.0 # reward term named {d}
                        # wet a dry fern:
                        self.fern_wet = True
                        self.environment_just_changed = True
                elif (self.row,self.col) == self.__chg__:
                    reward += -20.0     # reward term named {e}
                else:
                    reward += - 2.0     # reward term named {f}

        # charging
        if ENABLE_CHARGE:
            if (self.row,self.col) == self.__chg__:
                if self.charge==3:
                    reward += - 0.5     # reward term named {g}
                self.charge = min(3, self.charge+1) 
                if self.has_water:
                    reward += - 2.0     # reward term named {h}
            elif (dr or dc) and coin(self.PROB_LOSE_CHARGE):
                self.charge = max(0, self.charge-1)
            #
            if self.charge == 0:
                reward += - 1.5         # reward term named {i}

        # door
        if ENABLE_DOOR:
            if not self.door_open and coin(self.PROB_DOOR_OPENS):
                self.door_open = True
                self.environment_just_changed = True
            elif self.door_open and coin(self.PROB_DOOR_CLOSES):
                if (self.row,self.col) == self.__dor__:
                    reward += -10.0     # reward term named {j}
                else:
                    self.door_open = False
                    self.environment_just_changed = True

        # locomotion
        # affects (self.row, self.col), so paragraph should be placed last!
        if dr or dc:
            reward += - 0.1             # reward term named {k}
        r, c = (self.row+dr, self.col+dc) 
        if self.in_bounds(r,c):
            self.row, self.col = (r,c)
        else:
            reward += - 0.5             # reward term named {l}

        self.steps_elapsed += 1 
        self.total_reward = reward + self.GAMMA * self.total_reward  
        return reward


    def recent_change(self):
        return self.environment_just_changed

    def reset(self):
        (self.row, self.col, self.has_water, self.charge,
         self.fern_wet, self.bucket_full, self.door_open) = random_el(self.STATES)
        self.steps_elapsed = 0 
        self.total_reward = 0
        self.environment_just_changed = True

    def print_verbose(self):
        charge = round(33.33*self.charge)
        reward = round(self.total_reward, 2)

        display('<b>')
        display('steps so far: <y>{:5d}<b>; '   , self.steps_elapsed                    )
        display('total reward: <y>{:8s}<b>; ' , ('<r>' if reward<0 else '')+str(reward)+' '*2)
        display('\n')
        display('robot <y>{:5s}<b> has water; '  , "<r>ISN'T" if self.has_water else 'is' )
        display('robot has charge <y>{:3s}<b>; '   , ('<r>' if charge<=33 else '')+str(charge))
        display('\n')
        display('fern is <y>{:3s}<b>; '         , 'ok' if self.fern_wet else '<r>DRY')
        display('bucket is <y>{:5s}<b>; '       , 'full' if self.bucket_full else '<r>EMPTY')
        display('door is <y>{:6s}<b>; '         , 'open' if self.door_open else '<r>CLOSED')
        display('\n')
        display('<a>')

        display('<b>.'+'---.'*6+'---.<a>\n')
        for r in range(7):

            display('<b>|<a>')

            for c in range(7):
                cell, color = (
                    (' # ', '<b>')                                          if (r,c) in self.__walls__ else
                   (('FRN', '<e>') if self.fern_wet    else ('frn', '<y>')) if (r,c) == self.__frn__ else 
                    ('chg', '<y>')                                          if (r,c) == self.__chg__ else 
                   (('BKT', '<c>') if self.bucket_full else ('bkt', '<y>')) if (r,c) == self.__bkt__ else 
                    ('h20', '<c>')                                          if (r,c) == self.__h20__ else 
                   (('[ ]', '<a>') if self.door_open   else ('[=]', '<m>')) if (r,c) == self.__dor__ else 
                    ('   ', '<a>')
                )
                cell = [color+c+GRAY for c in cell]

                if (r,c) == (self.row, self.col):
                    cell[1] = '<c>@<a>' if self.has_water else '<r>O<a>'

                display(''.join(cell))
                display('|' if c+1!=7 else '<b>|<a>')

            display('\n')

            if r+1 != 7: display('<b>:<a>'+'---:'*6+'---<b>:<a>\n')
            else:        display("<b>'"+"---'"*6+"---'<a>\n")

    def reset_cursor_after_print_verbose(self):
        display('\033[1A'*(3+(1+2*7)+1) + '\033[1D'*90)

    def in_bounds(self, r,c, door=None):
        return ((0<=r<7 and 0<=c<7) and 
                ((r,c) not in self.__walls__) and 
                (((r,c) != door) if door is not None else
                  (self.door_open or (r,c)!=self.__dor__)
                )
               )


#==============================================================================
#===  4. USING AND LEARNING POLICIES FOR THE MDP  =============================
#==============================================================================

def simulate_policy(world, policy, nb_steps):
    world.reset()
    for _ in range(nb_steps):
        state = world.state()
        action = policy(state)
        world.perform_action(action)
    return world.total_reward

def q_learn(world, nb_lives, nb_steps_per_life, learn_rate,
            exploration_policy, epsilons = [2**-7, 2**-5, 2**-3, 2**-1],
            optimism = 1.0,
            curriculum=True,
            pseudo_goal_bonus = 1.0, nb_pseudo_goals = 20,
            nb_buffer_updates = 2, buffer_prob = 0.01, buffer_size = 1000, 
           ):

    q_main    = QFunction(world.STATES, world.ACTIONS, optimism)
    q_explore = QFunction(world.STATES, world.ACTIONS)

    experience_buffer = []

    for l in tqdm.tqdm(range(nb_lives)):
        eps = np.random.choice(epsilons)
        q_explore.copy_from(q_main)

        pseudo_goals = {random_el(list(world.STATES))
                        for _ in range(nb_pseudo_goals)}

        world.reset()

        for t in range(nb_steps_per_life):
            s = world.state()
            a = (exploration_policy(s) if coin(eps) else
                 q_explore.query_a(s)                   )
            r = world.perform_action(a)

            if curriculum:
                if W.has_water: r += 0.2 
                if W.charge >= 2  : r += 0.2 

            new_state = world.state()

            # update buffer
            if coin(buffer_prob):
                experience_buffer.append((s, a, r, new_state))
            if len(experience_buffer) > buffer_size:
                experience_buffer = experience_buffer[1:]

            # update return q from experience
            q_main.update(s, a, r + world.GAMMA*q_main.query_aq(new_state), learn_rate)

            # update exploration q from experience
            if new_state in pseudo_goals:
                r += pseudo_goal_bonus
            q_explore.update(s, a, r + world.GAMMA*q_explore.query_aq(new_state), learn_rate)

            # update return q from buffer
            for _ in range(min(nb_buffer_updates, len(experience_buffer))):
                (s, a, r, new_state) = random_el(experience_buffer)
                q_main.update(s, a, r + world.GAMMA*q_main.query_aq(new_state), learn_rate)

    return q_main

#==============================================================================
#===  5. MAIN LOOP: LEARNING AND SIMULATION  ==================================
#==============================================================================

np.random.seed(0)

W = World()
A = list(W.ACTIONS)
S = list(W.STATES)
display('<y>{}<b> many actions<a>\n', len(A))
display('<y>{}<b> many states <a>\n', len(S))

def uniform_policy(state):
    return A[np.random.choice(len(A))]
#q_table = q_learn(W,1000,1000,
#                  learn_rate=0.1, exploration_policy=uniform_policy)
#np.save('testing.npy', q_table)
#q_table = np.load('q_table-1000-1000.npy', allow_pickle=True).item()
q_table = np.load('testing.npy', allow_pickle=True).item()

def policy_from_table(state):
    return q_table.query_a(state)

def flipbook(policy=uniform_policy):
    display('\n<c>INSTRUCTIONS: press ENTER to simulate one time step);')
    display('\n<c>DOT-THEN-ENTER, to simulate until non-robot-state changes\n\n')
    W.reset()
    W.print_verbose()
    W.reset_cursor_after_print_verbose()
    def step():
        a = policy(W.state()) #if coin(0.99) else uniform_policy(W.state()) 
        W.perform_action(a)
        W.print_verbose()
        display('<b>most recent action: move <y>{}<b> and <y>{}<b> water action'+' '*20,
                {(0,0):'nowhere', (0,-1):'left', (0,+1):'right', (+1,0):'down', (-1,0):'up'}[a[0]],
                {True:'perform', False:'no'}[a[1]])
        W.reset_cursor_after_print_verbose()
    while True:
        display('<r>')
        s = input('')
        step()
        if s=='.':
            while not W.recent_change():
                display('\n')
                step()
                time.sleep(0.02)

flipbook(policy=policy_from_table)
#flipbook(policy=uniform_policy)
