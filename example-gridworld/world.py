''' author: samtenka
    change: 2022-08-18
    create: 2022-08-07

    descrp: Learn (from reinforcement) to water house plants in a simulated
            world.  Interactively simulate a uniform random policy or a policy
            learned via q-learning --- the latter we may save to or load from
            files so as not to train redundantly.

            The agent may move across the house's fixed floorplan and may
            interact with various devices in the house including an h20 source,
            a fern, and a charging station.  The agent is rewarded for carrying
            water from the h20 source to the fern whenever the fern becomes
            dry.  See section 1 for MDP details.

            We allow experiments with different q-function approximators and
            q-learning times; with different special techniques to accelerate
            learning by encouraging exploration; and with different world 
            dynamics.  See sections 0.1.1, 0.1.2, and 0.1.3 for control knobs.

            See comments marked with the three-character token "(!)" for
            highlights (to study or to change) in the code below.

    to use: A typical use of this program involves 4 steps:
                (a) adjust loading and training parameters in section 0.1
                (b) run the command line command `python3 world.py` 
                (c) if training: wait for the progress bar to complete
                (d) observe the learned behavior by repeatedly pressing ENTER;

            Regarding (a): we may decide which policy to simulate by changing
            the value of POLICY_FROM in section 0.1.0: 'train' means we train a
            policy from scratch according to the indicated training settings;
            'load' means we load a policy previously trained using the
            indicated training settings; and 'uniform' means we use a uniform
            random policy, which doesn't need training.

            Training settings are indicated in sections 0.1.1, 0.1.2, 0.1.3.

            Here is a sequence of suggested experiments, each of which
            describes what to do in step (a) before we perform steps (bcd).  In
            all but experiments [0,2], we'll set POLICY_FROM='train'.  In each
            experiment, we assume that the training settings not mentioned are
            at their default values.
                [0] set POLICY_FROM='uniform' (so no training)
                [1] set POLICY_FROM='train' (so train with default values) 
                [2] set POLICY_FROM='load' to see how loading works 
                [3] set ENABLE_CHARGE=True --- do you observe new behavior? 
                [4] for ENABLE_CHARGE=True, is NB_LIVES=500 enough traintime?
            In [5,6,7,8,9], we enable both CHARGE and DOOR:
                [5] --- does the learned behavior seem optimal?
                [6] what if we set NB_LIVES=5000?  does this improve behavior?
                [7] what if we set OPTIMISM=True, instead?
                [8] what if we set CURRICULUM=True, instead?
                [9] what if we do [5,6,7]'s changes simultaneously?

            Regarding (c): setting NB_LIVES = 1000 and NB_STEPS_PER_LIFE = 1000
            should lead to a training progress bar that finishes in ~10 seconds
            with all other parameters at their default value (section 1.3
            describes these default values).  Five other parameters much affect
            training walltime:
                Q_FUNCTION='linear'     -->  x 6.   training time
                REPLAY=True             -->  x 2.   training time 
                ENABLE_BUCKET=True      -->  x 1.5  training time 
                ENABLE_CHARGE=True      -->  x 1.5  training time 
                ENABLE_DOOR  =True      -->  x 1.5  training time 

            Regarding (d): as described in the instructions that print when we
            run this script, we may type a fullstop before pressing ENTER in
            order to skip ahead in time to the next interesting state.  We may
            also force a desired action rather than following the chosen policy
            by typing one-character or two-character strings such as 
                <
            or
                ^
            or
                >w
            or
                xw
            before pressing ENTER.  Those four example commands respectively:
            move left; move up; move right while impelling or expelling water;
            stay put while impelling or expelling water.
'''

#==============================================================================
#===  0. GLOBAL CONSTANTS  ====================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.0. Imported Modules  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tqdm
import time
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.1. Simulation Parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#            \    /        \    /        \    /     
#             \  /          \  /          \  /      
#              \/            \/            \/       
#                                                   
#   (!)     I INVITE YOU TO MODIFY THIS SECTION!    
#                                                   
#              /\            /\            /\       
#             /  \          /  \          /  \      
#            /    \        /    \        /    \     

#---  0.1.0. should we train or load our policy?  or set it to uniform?  ------

POLICY_FROM = 'train'      #   'train' or 'load' or 'uniform'

#---  0.1.1. the q-function approximator and how long to train it  ------------

Q_FUNCTION = 'tabular'      #   'tabular' or 'linear'   --- default 'tabular'
NB_LIVES          = 1000    #   any nonnegative integer --- default 1000
NB_STEPS_PER_LIFE = 1000    #   any nonnegative integer --- default 1000

#---  0.1.2. enable or disable exploration boosting techniques  ---------------

EPSILON       = True        #   True or False   --- default True 
OPTIMISM      = False       #   True or False   --- default False
CURRICULUM    = False       #   True or False   --- default False
PSEUDOGOAL    = False       #   True or False   --- default False
REPLAY        = False       #   True or False   --- default False

#---  0.1.3. enable or disable complexities of the world ----------------------

ENABLE_BUCKET = False       #   True or False   --- default False
ENABLE_CHARGE = False       #   True or False   --- default False
ENABLE_DOOR   = False       #   True or False   --- default False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.2. Parameter-Based Strings  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FILE_NAME = 'q.{}-{}-{}.{}-{}-{}.{}-{}-{}-{}-{}.npy'.format(
                                                         int(ENABLE_BUCKET),
                                                         int(ENABLE_CHARGE),
                                                         int(ENABLE_DOOR  ),
                                                         Q_FUNCTION       ,
                                                         NB_LIVES         ,
                                                         NB_STEPS_PER_LIFE,
                                                         int(EPSILON   ),
                                                         int(OPTIMISM  ),
                                                         int(CURRICULUM),
                                                         int(PSEUDOGOAL),
                                                         int(REPLAY    ),
                                                        ) 

LQ = len(str(Q_FUNCTION       )) 
LL = len(str(NB_LIVES         ))
LS = len(str(NB_STEPS_PER_LIFE))

EXPLANATION_OFFSET = len('tricks  --(     ')
EXPLANATION = (
'''
             __ | | | {A}| {B}| {C}|  | | | | |
training    /   | | | {A}| {B}| {C}|  | | | | +- {}replay<b>
tricks     /    | | | {A}| {B}| {C}|  | | | +- {}pseudogoal<b>
to help --(     | | | {A}| {B}| {C}|  | | +- {}curriculum<b>
explore    \    | | | {A}| {B}| {C}|  | +- {}optimism<b>
            \__ | | | {A}| {B}| {C}|  +- {}epsilon<b>
             __ | | | {A}| {B}| {C}|                         
training    /   | | | {A}| {B}| {C}+- <c>{}<b> steps per life
time &  ---(    | | | {A}| {B}+- <c>{}<b> lives of training
model       \__ | | | {A}+- <c>{}<b> Q-function approximation
             __ | | |                                          
training    /   | | +- {}humans might open or close the door<b>
world's ---(    | +- {}robot needs occasional recharge<b>                
complexities\__ +- {}bucket can be filled and drained<b>                  
'''.replace('{A}', ' '*int(LQ/2.0))
   .replace('{B}', ' '*(int((LQ-1.0)/2.0) + int(LL/2.0)))
   .replace('{C}', ' '*(int((LL-1.0)/2.0) + int(LS/2.0)))
   .format(
       '<c>' if REPLAY          else '<a>',
       '<c>' if PSEUDOGOAL      else '<a>',
       '<c>' if CURRICULUM      else '<a>',
       '<c>' if OPTIMISM        else '<a>',
       '<c>' if EPSILON         else '<a>',
       NB_STEPS_PER_LIFE,
       NB_LIVES,
       Q_FUNCTION,
       '<c>' if ENABLE_DOOR     else '<a>',
       '<c>' if ENABLE_CHARGE   else '<a>',
       '<c>' if ENABLE_BUCKET   else '<a>',
       )        
)               
                
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.3. Fancy Printing: Colors and Cursor Motions  ~~~~~~~~~~~~~~~~~~~~~~~~~

GRAY   = '\033[30m' 
RED    = '\033[31m' 
GREEN  = '\033[32m' 
YELLOW = '\033[33m' 
BLUE   = '\033[34m' 
MAGENTA= '\033[35m' 
CYAN   = '\033[36m' 

UP     = '\033[1A' 
DOWN   = '\033[1B' 
RIGHT  = '\033[1C' 
LEFT   = '\033[1D' 

COLORS_BY_TAGS = {
  'a': GRAY   , 
  'r': RED    , 
  'e': GREEN  , 
  'y': YELLOW , 
  'b': BLUE   , 
  'm': MAGENTA, 
  'c': CYAN   , 
  'up'   : UP   , 
  'down' : DOWN , 
  'right': RIGHT, 
  'left' : LEFT , 
}

def display(s, *args):
    s = s.format(*args)
    for k,v in COLORS_BY_TAGS.items():
        s = s.replace('<'+k+'>', v)
    print(s, end='')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  0.4. Randomness Helpers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def init_random_number_generator():
    np.random.seed(0)

def coin(p):
    return np.random.random() < p

def rand_el(L):
    return L[np.random.choice(len(L))]

#==============================================================================
#===  1. OVERVIEW OF MARKOV DECISION PROCESS  =================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  1.0. States of the World  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        {holding water, not holding water} x
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
#~~~  1.1. Actions and Dynamics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
#~~~  1.2. Rewards  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    The robot experiences a reward that is a sum of several terms: 
        {a} [medium]    like when the fern is wet
        {b} [medium] dislike when the fern is dry

        {c} [ mild ] dislike of (im/ex)pelling water (motors sound annoying)
        {d} [STRONG] dislike of watering an already wet fern
        {e} [medium]    like of watering a dry fern
        {f} [STRONG] dislike of spilling water on charger
        {g} [medium] dislike of spilling water anywhere but fern or bucket

        {h} [ mild ] dislike of overcharging once full of charge
        {i} [medium] dislike of charging while holding water
        {j} [STRONG] dislike when no charge left (backup battery expensive)

        {k} [STRONG] dislike of blocking door from being closed

        {l} [ mild ] dislike of moving at all (motors make annoying sound)
        {m} [medium] dislike of bumping into the wall or into a closed door

    The discount factor is GAMMA=0.98, meaning that the robot cares most about
    its rewards within the next 50=1/(1-GAMMA) timesteps.  
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  1.3. Simplifications  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
    By default, we work in a simpler world wherein:
    -- the door is ALWAYS open and never closes
    -- the bucket is ALWAYS empty and can never be filled
    -- the robot's charge is ALWAYS 100% and the charging station has no effect

    In this simpler world, the robot just needs to learn to ferry water from
    the h20 source to the fern whenever the fern becomes dry.

    We may relax these constraints and allow the door, bucket, and/or charge
    to change over time by setting ENABLE_CHARGE, ENABLE_DOOR, ENABLE_BUCKET
    to True.
'''

#==============================================================================
#===  2. IMPLEMENTING THE MDP  ================================================
#==============================================================================

class World:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~  2.0. Initialization   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

        self.STATES = [(r,c,has_water,charge,fern_wet,buck_full,door_open)
                       for has_water in {True,False} 
                       for charge in (range(4) if ENABLE_CHARGE else [3])
                       for fern_wet in {True,False} 
                       for buck_full in ({True,False} if ENABLE_BUCKET else {False})
                       for door_open in ({True,False} if ENABLE_DOOR else {True})
                       for r in range(7)
                       for c in range(7) if self.in_bounds(r,c,door=door_open)
                      ]
        self.reset()

    def reset(self):
        (self.row, self.col, self.has_water, self.charge,
         self.fern_wet, self.buck_full, self.door_open) = rand_el(self.STATES)
        self.steps_elapsed = 0 
        self.total_reward = 0
        self.environment_just_changed = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~  2.1. Query State  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def state(self):
        return (self.row, self.col, self.has_water, self.charge,
                self.fern_wet, self.buck_full, self.door_open)

    def in_bounds(self, r,c, door=None):
        return ((0<=r<7 and 0<=c<7) and 
                ((r,c) not in self.__walls__) and 
                (((r,c) != door) if door is not None else
                  (self.door_open or (r,c)!=self.__dor__)
                )
               )

    def recent_change(self):
        return self.environment_just_changed

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~  2.2. Dynamics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
            reward += - 0.1               # reward term named {c}
            if (self.row,self.col) == self.__h20__:
                self.has_water = not self.has_water
                self.environment_just_changed = True
            elif ENABLE_BUCKET and (self.row,self.col) == self.__bkt__:
                # swap contents:
                self.has_water, self.buck_full = self.buck_full, self.has_water
                self.environment_just_changed = True
            elif self.has_water:
                self.has_water = False
                self.environment_just_changed = True
                if (self.row,self.col) == self.__frn__:
                    if self.fern_wet:
                        reward += -10.0     # reward term named {d}
                    else:
                        # wet a dry fern:
                        self.fern_wet = True
                        self.environment_just_changed = True
                        reward += + 2.0     # reward term named {e}
                elif (self.row,self.col) == self.__chg__:
                    reward += -20.0         # reward term named {f}
                else:
                    reward += - 2.0         # reward term named {g}

        # charging
        if ENABLE_CHARGE:
            if (self.row,self.col) == self.__chg__:
                if self.charge==3:
                    reward += - 0.5         # reward term named {h}
                self.charge = min(3, self.charge+1) 
                self.environment_just_changed = True
                if self.has_water:
                    reward += - 2.0         # reward term named {i}
            elif (dr or dc) and coin(self.PROB_LOSE_CHARGE):
                self.charge = max(0, self.charge-1)
                self.environment_just_changed = True
            #
            if self.charge == 0:
                reward += - 1.5             # reward term named {j}

        # door
        if ENABLE_DOOR:
            if not self.door_open and coin(self.PROB_DOOR_OPENS):
                self.door_open = True
                self.environment_just_changed = True
            elif self.door_open and coin(self.PROB_DOOR_CLOSES):
                if (self.row,self.col) == self.__dor__:
                    reward += -10.0         # reward term named {k}
                else:
                    self.door_open = False
                    self.environment_just_changed = True

        # locomotion (affects (row, col), so paragraph should be placed last!)
        if dr or dc:
            reward += - 0.1                 # reward term named {l}
        r, c = (self.row+dr, self.col+dc) 
        if self.in_bounds(r,c):
            self.row, self.col = (r,c)
        else:
            reward += - 0.5                 # reward term named {m}

        self.steps_elapsed += 1 
        self.total_reward = reward + self.GAMMA * self.total_reward  
        return reward

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~  2.3. Printing  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def str_from_cell(self, r, c):
        cell, color = (
            (' # ', '<b>')                                        if (r,c) in self.__walls__ else
           (('FRN', '<e>') if self.fern_wet  else ('frn', '<y>')) if (r,c) == self.__frn__ else 
            ('chg', '<y>')                                        if (r,c) == self.__chg__ else 
           (('BKT', '<c>') if self.buck_full else ('bkt', '<y>')) if (r,c) == self.__bkt__ else 
            ('h20', '<c>')                                        if (r,c) == self.__h20__ else 
           (('[ ]', '<a>') if self.door_open else ('[=]', '<m>')) if (r,c) == self.__dor__ else 
            ('   ', '<a>')
        )
        cell = [color+c+GRAY for c in cell]

        if (r,c) == (self.row, self.col):
            cell[1] = '<c>@<a>' if self.has_water else '<r>O<a>'

        return ''.join(cell)

    def print_status(self):
        charge = round(33.33*self.charge)
        charge = ('<r>' if charge<=33 else '')+str(charge)

        reward = self.total_reward
        reward = ('<r>' if reward<0 else '')  +'${:+.2f}'.format(reward)

        water  = 'is' if self.has_water else "<r>ISN'T"
        fern   = 'ok' if self.fern_wet else '<r>DRY'
        bucket = 'full' if self.buck_full else '<r>EMPTY'
        door   = 'open' if self.door_open else '<r>CLOSED'

        display('<b>')
        display('<c>{:5d}<b> steps taken; '         , self.steps_elapsed)
        display('total reward: <c>{:8s}<b>    '     , reward            )
        display('\n')
        display('robot <c>{:5s}<b> holding water; ' , water             )
        display('robot has charge <c>{:3s}<b>    '  , charge            )
        display('\n')
        display('fern is <c>{:3s}<b>; '             , fern              )
        display('bucket is <c>{:5s}<b>; '           , bucket            )
        display('door is <c>{:6s}<b>    '           , door              )
        display('\n')
        display('<a>')

    def print_verbose(self):
        self.print_status()

        display('<b>.'+'---.'*6+'---.<a>\n')
        for r in range(7):
            display('<b>|<a>')
            for c in range(7):
                display(self.str_from_cell(r,c))
                display('|' if c+1!=7 else '<b>|<a>')
            if r+1 != 7: display('\n<b>:<a>'+'---:'*6+'---<b>:<a>\n')
            else:        display("\n<b>'"+"---'"*6+"---'<a>\n")

    def reset_cursor_after_print_verbose(self):
        display('<up>'*(3+(1+2*7)+1) + '<left>'*90)

    def get_action_name(self, a):
        motion = {(0,0):'nowhere',
                  (0,-1):'left'  ,
                  (0,+1):'right' ,
                  (+1,0):'down'  ,
                  (-1,0):'up'    ,}[a[0]] 
        water  =  {True:'perform',
                   False:'no'    ,}[a[1]]
        return 'move <c>{}<b> and <c>{}<b> water action'.format(motion, water)

#==============================================================================
#===  2. Q-FUNCTIONS AS DATA STRUCTURES  ======================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  3.0. Q-Function Template ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QFunction:
    def __init__(self, STATES, ACTIONS, optimism=0.0): pass

    def copy_from(self, rhs): pass

    #            \    /        \    /        \    /     
    #             \  /          \  /          \  /      
    #              \/            \/            \/       
    #                                                   
    #   (!)     KEY LINES OF CODE FOR ALL QLEARNING: 
    #                                                   
    #              /\            /\            /\       
    #             /  \          /  \          /  \      
    #            /    \        /    \        /    \     
    def update(self, s, a, r, new_state, gamma, learn_rate):
        new_val = r + gamma * self.query_aq(new_state) 
        self.nudge_toward(s, a, new_val, learn_rate)

    def nudge_toward(self, s, a, new_val, learn_rate): pass

    def query_q(self, s, a): pass

    def query_a(self, s):
        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
        return a
    def query_aq(self, s):
        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
        return q

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  3.1. Tabular Q-Function  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QFunctionTabular(QFunction):
    def __init__(self, STATES, ACTIONS, optimism=0.0):
        self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}
        self.ACTIONS = ACTIONS

    def copy_from(self, rhs):
        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
        self.ACTIONS = rhs.ACTIONS

    #            \    /        \    /        \    /     
    #             \  /          \  /          \  /      
    #              \/            \/            \/       
    #                                                   
    #   (!)     KEY LINE OF CODE FOR TABULAR QLEARN: 
    #                                ^^^^^^^            
    #              /\            /\            /\       
    #             /  \          /  \          /  \      
    #            /    \        /    \        /    \     
    def nudge_toward(self, s, a, new_val, learn_rate): 
        self.q_table[(s,a)] += learn_rate * (new_val - self.q_table[(s,a)])

    def query_q(self, s, a):
        return self.q_table[(s,a)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  3.1. Linear Q-Function  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QFunctionLinear(QFunction):
    def __init__(self, STATES, ACTIONS, optimism=0.0):

        #self.q_table = {(s,a):optimism for s in STATES for a in ACTIONS}
        self.q_table = {(s,):optimism for s in STATES}

        self.rcs = {tuple(s[:2]) for s in STATES}
        example_state = next(iter(STATES))
        self.state_axes = [set([]) for _ in example_state[2:]] 
        for s in STATES:
            for i,x in enumerate(s[2:]):
                self.state_axes[i].add(x)
        self.q_aux = [{(rc,x,a):0.0 for rc in self.rcs
                                    for x in axis
                                    for a in ACTIONS  }
                      for axis in self.state_axes      ]

        self.ACTIONS = ACTIONS

    def copy_from(self, rhs):
        self.q_table = {k:rhs.q_table[k] for k in rhs.q_table}
        self.q_aux   = [{k:d[k] for k in d} for d in rhs.q_aux]
        self.ACTIONS = rhs.ACTIONS

    #            \    /        \    /        \    /     
    #             \  /          \  /          \  /      
    #              \/            \/            \/       
    #                                                   
    #   (!)     KEY LINES OF CODE FOR LINEAR QLEARN: 
    #                                 ^^^^^^            
    #              /\            /\            /\       
    #             /  \          /  \          /  \      
    #            /    \        /    \        /    \     
    def nudge_toward(self, s, a, new_val, learn_rate, aux_factor=0.1):
        discrepancy = new_val - self.query_q(s,a) 
        #self.q_table[(s,a)] += learn_rate * discrepancy
        self.q_table[(s,)] += learn_rate * discrepancy
        for i,x in enumerate(s[2:]):
            self.q_aux[i][(tuple(s[:2]),x,a)] += learn_rate*(aux_factor * discrepancy)

    def query_q(self, s, a):
        #return self.q_table[(s,a)] + sum(self.q_aux[i][(tuple(s[:2]),x,a)]
        return self.q_table[(s,)] + sum(self.q_aux[i][(tuple(s[:2]),x,a)]
                                        for i,x in enumerate(s[2:]))
    def query_a(self, s):
        _, a = max((self.query_q(s,a),a) for a in self.ACTIONS)
        return a

    def query_aq(self, s):
        q, _ = max((self.query_q(s,a), a) for a in self.ACTIONS)
        return q

#==============================================================================
#===  4. USING AND LEARNING POLICIES FOR THE MDP  =============================
#==============================================================================

def uniform_policy(state):
    global A
    return rand_el(A)

def policy_from_table(q_table):
    def policy(state):
        return q_table.query_a(state)
    return policy

def simulate_policy(world, policy, nb_steps):
    world.reset()
    for _ in range(nb_steps):
        state = world.state()
        action = policy(state)
        world.perform_action(action)
    return world.total_reward

def q_learn(world, nb_lives, nb_steps_per_life, learn_rate,
            explore_policy=uniform_policy, epsilons=[2**-7,2**-5,2**-3,2**-1],
            optimism =  2.0,
            curriculum=True,
            pg_bonus = 1.0, nb_pgs = 20,
            nb_replays = 2, mem_prob = 0.01, nb_mem = 1000, 
           ):

    QF = {'tabular':QFunctionTabular, 'linear':QFunctionLinear}[Q_FUNCTION]
    q_main    = QF(world.STATES, world.ACTIONS, optimism)
    q_explore = QF(world.STATES, world.ACTIONS)

    experience_buffer = []

    for l in tqdm.tqdm(range(nb_lives)):
        lr = learn_rate # * (0.1*nb_lives) / (1 + 0.1*nb_lives + l) 

        eps = rand_el(epsilons)
        q_explore.copy_from(q_main)

        pseudo_goals = {rand_el(list(world.STATES))
                        for _ in range(nb_pgs)}

        world.reset()

        for t in range(nb_steps_per_life):
            # perform an action and hence gain one step more of experience: 
            s = world.state()
            if coin(eps): a = explore_policy(s)
            else        : a = q_explore.query_a(s)
            r = world.perform_action(a)
            new_state = world.state()

            #   ...add artificial incentives to accelerate learning:
            if curriculum:
                if W.has_water  : r += 0.2 
                if W.charge >= 2: r += 0.2 
                if W.buck_full  : r += 0.2 

            # update memory buffer:
            if coin(mem_prob):
                experience_buffer.append((s, a, r, new_state))
            if len(experience_buffer) > nb_mem:
                experience_buffer = experience_buffer[1:]

            # update main Q from experience:
            #            \    /        \    /        \    /     
            #             \  /          \  /          \  /      
            #              \/            \/            \/       
            #                                                   
            #   (!)     THIS IS THE ESSENTIAL LEARNING LINE:    
            #                                                   
            #              /\            /\            /\       
            #             /  \          /  \          /  \      
            #            /    \        /    \        /    \     
            q_main.update(s, a, r, new_state, world.GAMMA, lr)

            # update exploration Q from experience:
            #   ...add artificial incentives to accelerate learning:
            if new_state in pseudo_goals:
                r += pg_bonus
            q_explore.update(s, a, r, new_state, world.GAMMA, lr)

            # update main Q from memory buffer:
            for _ in range(min(nb_replays, len(experience_buffer))):
                (s, a, r, new_state) = rand_el(experience_buffer)
                q_main.update(s, a, r, new_state, world.GAMMA, lr)

    return q_main

#==============================================================================
#===  5. MAIN LOOP: LEARNING AND SIMULATION  ==================================
#==============================================================================

def step(W, a):
    W.perform_action(a)
    W.print_verbose()
    display('<b>most recent action: {}'+' '*20, W.get_action_name(a))
    W.reset_cursor_after_print_verbose()

def simulate(W, policy=uniform_policy):
    display('\n<b>INSTRUCTIONS: enter an <y>empty string<b> to step once;')
    display('\nenter <y>.<b> to simulate until non-location-state changes;')
    display('\nenter <y><<b> or <y>><b> or <y>^<b> or <y>v<b> (or <y>x<b>) to force a (non)move,')
    display('\nappending <y>w<b> to perform a water action as well.\n\n')
    W.reset()
    W.print_verbose()
    W.reset_cursor_after_print_verbose()
    while True:
        display('<y>')
        s = input('')
        if s and s[0] in '<>^vx': 
            a = ({'<':(0,-1),'>':(0,+1),'^':(-1,0),'v':(+1,0),'x':(0,0)}[s[0]],
                 (len(s)>=2 and s[1]=='w')) 
            step(W, a)
        else:
            step(W, policy(W.state()))
        if s!='.': continue
        while not W.recent_change():
            display('\n')
            step(W, policy(W.state()))
            time.sleep(0.02)
        #print(q_table.q_aux)

if __name__=='__main__':
    init_random_number_generator()
    
    W = World()
    A = list(W.ACTIONS)
    S = list(W.STATES)
    display('<b>the world allows <c>{}<b> actions ', len(A))
    display('<b>in each of <c>{}<b> many states\n', len(S))
    
    if POLICY_FROM=='train': 
        display('<b>using q-learning to train <c>{}<b>...', FILE_NAME)
        display(EXPLANATION.replace('\n', '\n'+' '*(len('using q-learning to train q.')-EXPLANATION_OFFSET)))
        kw = {}
        if not EPSILON   :  kw['epsilons'] = [0.0]
        if not OPTIMISM  :  kw['optimism'] = 0.0
        if not CURRICULUM:  kw['curriculum'] = False
        if not PSEUDOGOAL:  kw['pg_bonus']=kw['nb_pgs'] = 0
        if not REPLAY    :  kw['nb_replays']=kw['mem_prob']=kw['nb_mem'] = 0
        q_table = q_learn(W,NB_LIVES,NB_STEPS_PER_LIFE,
                          learn_rate=0.1, **kw)
        display('<c>saving<b> Qfunction to <c>{}<b>\n', FILE_NAME)
        np.save(FILE_NAME, q_table)
        policy = policy_from_table(q_table)
        policy_name = FILE_NAME
    elif POLICY_FROM=='load':
        display('<c>loading<b> Qfunction from <c>{}<b>', FILE_NAME)
        display(EXPLANATION.replace('\n', '\n'+' '*(len('loading Qfunction from q.')-EXPLANATION_OFFSET)))
        q_table = np.load(FILE_NAME, allow_pickle=True).item()
        policy = policy_from_table(q_table)
        policy_name = FILE_NAME
    elif POLICY_FROM=='uniform':
        policy = uniform_policy
        policy_name = 'uniform'

    display('\nexecuting policy <c>{}<b>...\n', policy_name)
    simulate(W, policy=policy)
