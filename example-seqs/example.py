''' author: sam tenka
    change: 2022-07-13
    create: 2022-07-13
    descrp: .  
    depend: 
    jargon: we'll consistently use these abbreviations when naming variables:
                dec_func    --- decision function
                idx(s)      --- index/indices within list of all examples 
                nb_         --- number of (whatever follows the underscore)
                side        --- sidelength of image, measured in pixels
                x           --- raw input vector
                y           --- bit-valued label
                vert        --- to do with a graph's vertical axis
                hori        --- to do with a graph's horizontal axis
    thanks: featurization idea inspired by abu-mostafa's book
    to use: Run `python3 example.py`.
'''

#===============================================================================
#==  0. PREAMBLE  ==============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.0. universal constants  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.0.0. import modules  ----------------------------------------

from matplotlib import pyplot as plt                                            
import numpy as np                                                              
import tqdm
#from process_reviews import MAX_WORD_COUNT, get_lines, get_shared_words, featurize
from tablet_data import ALPHABET, make_paragraph


#--------------  0.2.1. colors  ------------------------------------------------

WHITE        = np.array([1.0 ,1.0 ,1.0 ])
SMOKE        = np.array([ .9 , .9 , .9 ])
SLATE        = np.array([ .5 , .5 , .5 ])
SHADE        = np.array([ .1 , .1 , .1 ])
BLACK        = np.array([ .0 , .0 , .0 ])

RED          = np.array([1.0 , .0 , .0 ]) #####
ORANGE       = np.array([ .75,0.25, .0 ]) #    
BROWN        = np.array([ .5 ,0.5 , .0 ]) ###    # i.e., dark YELLOW
OLIVE        = np.array([ .25,0.75, .0 ]) #    
GREEN        = np.array([ .0 ,1.0 , .0 ]) #####
AGAVE        = np.array([ .0 , .75, .25]) #    
CYAN         = np.array([ .0 , .5 , .5 ]) ###  
JUNIPER      = np.array([ .0 , .25, .75]) #    
BLUE         = np.array([ .0 , .0 ,1.0 ]) ##### 
INDIGO       = np.array([ .25, .0 , .75]) #    
MAGENTA      = np.array([ .5 , .0 , .5 ]) ###  
AMARANTH     = np.array([ .75, .0 , .25]) #    

def overlay_color(background, foreground, foreground_opacity=1.0):
    background += foreground_opacity * (foreground - background)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.1. global parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.1.0. data preparation ---------------------------------------

DATA_DIM     = len(ALPHABET)

NB_TEST      =  50 
NB_TRAIN_MAX =  500
NB_TRAIN     =  500

#--------------  0.1.1. model and training paremeters  -------------------------

LEAK = 0.1 
BIAS = True

#--------------  0.1.2. plotting and writing parameters  -----------------------

PLT_SIDE = 300
MARG     = 2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.2. global initialization and computations  ~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.2.0. parameterize randomness for replicability  -------------

np.random.seed(0)

#--------------  0.2.1. generate artificial data  ------------------------------

all_lines = [make_paragraph() for _ in range(NB_TRAIN_MAX+NB_TEST)]
print(min(len(s) for s in all_lines))

#--------------  0.2.2. format data  ------------------------------------------- 

char_by_idx = {i:c for i,c in enumerate(ALPHABET)}
idx_by_char = {c:i for i,c in char_by_idx.items()}
onehots = np.eye(len(ALPHABET))

#all_x = np.array([[onehots[idx_by_char[c]] for c in line[:80]] for line in all_lines])
all_x = np.array([[idx_by_char[c] for c in line[:80]] for line in all_lines])

#input(all_x[0][:10])

#--------------  0.2.3. shuffle and split  -------------------------------------

idxs = np.arange(len(all_x))
np.random.shuffle(idxs)
all_x = all_x[idxs]
#
train_idxs = np.arange(0           , NB_TRAIN            )
test_idxs  = np.arange(NB_TRAIN_MAX, NB_TRAIN_MAX+NB_TEST)

#===============================================================================
#==  1. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.0. Define Model  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# SIMPLE RNN MODEL
#   picture:  this RNN tries to predict the next symbol in a sequence of symbols
#
#   GENERAL THEMES IN DEEP LEARNING DESIGN:
#     --- symmetry : where a word is in a sentence doesn't affect its meaning (to leading order)
#                    where a sentence is in a book doesn't affect its meaning (to leading order)
#        --- address with `weight sharing`
#     --- locality : a word's meaning depends mostly on the nearby previous words 
#        --- sparse weight matrices, i.e.
#            no direct connections from far away words to current word being processed 
#
#   _______ _________________ _________________ _________________ _________________
#   t=4    |          t=3    |         t=2     |          t=1    |          t=0            
#          |      x          |      x          |      x          |      x        
#          |     /           |     /           |     /           |     /         
#               /C                /C                /C                /C         
#              /                 /                 /                 /           
#       h--#--z-----------h--#--z-----------h--#--z-----------h--#--z-----------h
#                   B    /            B    /            B    /            B    /
#                       /D                /D                /D                /D
#                      /                 /                 /                 /  
#                     o                 o                 o                 o   
#                    /                 /                 /                 /    
#                   #                 #                 #                 #     
#                  /                 /                 /                 /      
#                 p                 p                 p                 p       
#
#   parameters: 
#       A : h x 1   
#       B : h x h
#       C : h x d
#       D : d x h
#   forward model:
#       h_0 = A 
#       z_t = B h_t + C x_t
#       h_{t+1} = [sigma(z_t), 1]
#       o_t = D h_t
#       p_t = softmax(o_t)
#   loss:
#       l_t = -log(p_t)*x_t
#       l = sum_t l_t
sigma = lambda z: 1.0/(1.0+np.exp(z))
dsigma = lambda z: (lambda s: s*(1-s))(sigma(z))
# here, x is a single sentence (so a sequence of one-hot encoded symbols)
#def forward(weights, x): 
#    ''' goes through x and figures out neural net's 
#    prediction for x[5] based on x[0:5], etc'''
#    A,B,C,D = weights
#
#    # forward prop:
#    h = [None for _ in range(len(x))] 
#    z = [None for _ in range(len(x))] 
#    o = [None for _ in range(len(x))] 
#    p = [None for _ in range(len(x))] 
#    l = [None for _ in range(len(x))] 
#    h[0] = A
#    #t = 0
#    #while t != len(x): 
#    for t in range(len(x)):
#        z[t] = B.dot(h[t]) + C.dot(x[t])
#        o[t] = D.dot(h[t])
#        p[t] = np.exp(o[t]) / np.sum(np.exp(o[t])) 
#        h[t+1] = np.concatenate((sigma(z[t]), [1]))
#        l[t] = -np.log(p[t]).dot(x[t]) 
#    total_loss = np.mean(l)


# x = "The quick brow "
#                    ^
#         {'n': 0.99, 's':0.01, ...}
#
# x = "The quick brown fox jumped over  "
#                                      ^
#                            {'z': 0.01, 'm':0.2, 't':0.2, ...}
#
# x = "The quick brown fox jumped over many ..."
# x = "The quick brown fox jumped over the ..."
#def generate_sentence(weights, nb_symbols=80, MODE='RANDOM'): 
#    A,B,C,D = weights
#
#    # HOW DO WE BUILD x?
#    x = [None for _ in range(nb_symbols)]
#    h = [None for _ in range(nb_symbols)] 
#    z = [None for _ in range(nb_symbols)] 
#    o = [None for _ in range(nb_symbols)] 
#    p = [None for _ in range(nb_symbols)] 
#    l = [None for _ in range(nb_symbols)] 
#    h[0] = A
#    for t in range(nb_symbols):
#        # at this point we know h[t]
#        o[t] = D.dot(h[t])
#        p[t] = np.exp(o[t]) / np.sum(np.exp(o[t])) 
#        # at this point we know h[t], o[t], p[t]
#        #
#        if MODE=='RANDOM':
#            x[t] = np.random.choice(range(len(ALPHABET)), p[t])
#        elif MODE=='MAX':
#            x[t] = max((pr,i) for i,pr in enumerate(p[t]))[1] # argmax
#        # at this point we know h[t], o[t], p[t], x[t]
#        z[t] = B.dot(h[t]) + C.dot(x[t])
#        #
#        h[t+1] = np.concatenate((sigma(z[t]), [1]))
#        #
#    return x

# simple RNN

#make_dec_func_2 = lambda A,B: lambda x: np.dot(A, lrelu(np.matmul(B,x)))
#make_dec_func   = make_dec_func_2
#
#make_classifier = lambda dec_func: lambda x: 0 if dec_func(x)<=0 else 1
#
#is_correct = lambda classifier, idx: 1 if all_y[idx]==classifier(all_x[idx]) else 0
#
#error_rate = lambda classifier, idxs: np.mean([1.0-is_correct(classifier, idx) for idx in idxs])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.1. Train and Test Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.1.0. loss functions and gradients  --------------------------

#xent_softplus  = lambda z: np.log(1. + np.exp(z))
#dxent_softplus = lambda z: 0 if 1+z<0 else   1 
softmax = lambda o: (lambda e: e/np.sum(e))(np.exp(o))

def sample_from(weights, nb_chars=80):
    A,B,C,D = weights 
    hs = [0.0*A] # let's start with zero instead of A
    zs = []
    ps = []
    ls = []
    xs = []
    for _ in range(nb_chars):
        ps.append(softmax(D.dot(hs[-1])))
        xs.append(np.random.choice(range(DATA_DIM), p=ps[-1]))
        zs.append(B.dot(hs[-1]) + C[:,xs[-1]])
        #
        hs.append(np.concatenate((sigma(zs[-1]), [1.0])))
    return ''.join(char_by_idx[c] for c in xs)

def gradient(weights, x, l2_reg): 
    A,B,C,D = weights 

    # forward prop:
    hs = [0.0*A] # let's start with zero instead of A
    zs = []
    ps = []
    ls = []
    for c in x:
        ps.append(softmax(D.dot(hs[-1])))
        ls.append(np.log(1.0/ps[-1][c]))
        zs.append(B.dot(hs[-1]) + C[:,c])
        #
        hs.append(np.concatenate((sigma(zs[-1]), [1.0])))
    l = np.mean(ls)
           
    # back prop:
    dl_dA = l2_reg*A
    dl_dB = l2_reg*B
    dl_dC = l2_reg*C
    dl_dD = l2_reg*D

    ## DYNAMIC PROGRAMMING 
    #dl_dz_next = np.zeros(len(A)-1)#0.0*A
    ##for t in range(len(x))[::-1]:
    #for t in [len(x)-1-i for i in range(len(x))]:

    # 
    # 
    # 
    m = int(np.random.random()*(len(x)-10))
    M = m + 10 
 
    dl_dz_next = np.zeros(len(A)-1)#0.0*A
    #for t in range(len(x))[::-1]:
    for t in range(m,M)[::-1]:
        dl_do = ps[t] - onehots[x[t]]
        dl_dh = np.transpose(D).dot(dl_do) + np.transpose(B).dot(dl_dz_next)
        dl_dz = dsigma(zs[t]) * dl_dh[:-1] 
        #dl_dA += 
        #
        #  if v has shape (5,), w has shape (6,)
        #  then np.outer(v,w) has shape (5,6)
        #  np.outer(v,w)[i,j] == v[i]*w[j]
        #
        #  o = D*h
        #  o[i] = sum(D[i][j] * h[j] for j in range(dimension of h))
        #
        #   o[i] = a*b + c*s + e*f
        #
        #  how can we relate dl/dc to dl/do[i]?
        #       dl/dc = dl/do * do/dc = dl/do * s     
        #     
        #  how can we relate dl/dD[i][j] to dl/do[i]?
        #       dl/dD[i][j] = dl/do[i] * do[i]/dD[i][j] = dl/do[i] * h[j] 
        #
        dl_dB += np.outer(dl_dz_next, hs[t])
        dl_dC += np.outer(dl_dz_next, x[t])
        dl_dD += np.outer(dl_do, hs[t]) 
        dl_dz_next = dl_dz

    return (dl_dA,
            dl_dB,
            dl_dC,
            dl_dD, l)

def gradients(weights, idxs=train_idxs, l2_reg=0.01): 
    gs = [gradient(weights, all_x[i], l2_reg) for i in idxs]
    gAs = [g[0] for g in gs] 
    gBs = [g[1] for g in gs] 
    gCs = [g[2] for g in gs] 
    gDs = [g[3] for g in gs] 
    #
    ls  = [g[4] for g in gs] 
    return (np.mean(gAs, axis=0),
            np.mean(gBs, axis=0),
            np.mean(gCs, axis=0),
            np.mean(gDs, axis=0),
            np.mean(ls))

#--------------  1.1.2. gradient descent  --------------------------------------

def get_train_stats(weights, l2_reg):
    #dec_func = make_dec_func(*weights)
    #clsfier = make_clsfier(dec_func)
    _, _, _, _, l = gradients(weights, train_idxs, l2_reg)
    return {'train-loss':l,
            #'train-err':error_rate(clsfier, train_idxs),
            #'test-err' :error_rate(clsfier, test_idxs ), 
            }

# FOR SIMPLICITY, NO BIASES!!

def initialize_weight(m,n=None):
    #return ((np.random.randn(m,n+1) / np.sqrt(m+n+1)) if n is not None else
    return ((np.random.randn(m,n  ) / np.sqrt(m+n  )) if n is not None else
             np.random.randn(m    )                                        )

def gradient_descend(nb_hiddens, learning_rate,
                     nb_steps, report_every=None, batch_size=32, l2_reg=0.000,
                     weights = None):
    # initialize:
    A,B,C,D = weights if weights is not None else (
        initialize_weight(nb_hiddens+1),
        initialize_weight(nb_hiddens, nb_hiddens+1),
        initialize_weight(nb_hiddens, DATA_DIM  ),
        initialize_weight(DATA_DIM  , nb_hiddens+1),
    )
    print('training with '
            '\033[33m{:2d}\033[34m hiddens and '
            'learning rate \033[33m{:.4f}\033[34m...'.format(nb_hiddens, learning_rate))

    # main loop:
    for t in tqdm.tqdm(range(nb_steps)):
        # sgd update: 
        batch = np.random.choice(train_idxs, batch_size, replace=False)
        gA, gB, gC, gD, l = gradients((A,B,C,D), batch, l2_reg)
        A -= 1.0 * learning_rate * gA #* np.random.choice(2, gA.shape) 
        B -= 1.0 * learning_rate * gB #* np.random.choice(2, gB.shape)
        C -= 1.0 * learning_rate * gC #* np.random.choice(2, gC.shape)
        D -= 1.0 * learning_rate * gD #* np.random.choice(2, gD.shape)
         
        # report progress during training loop:
        if report_every is None or t%report_every: continue

        # print numeric scores:
        train_stats = get_train_stats((A,B,C,D), l2_reg)
        print('     after \033[33m{:6d}\033[34m steps: '
              'train-loss=\033[33m{:6.3f}\033[34m '
            #   'train-err=\033[33m{:5.1f}%\033[34m '
            #    'test-err=\033[33m{:5.1f}%\033[34m '
            .format(
            t,
            train_stats['train-loss'],
            #100*train_stats['train-err'],
            #100*train_stats['test-err'],
            ))
        print(sample_from((A,B,C,D)))

    return ((A,B,C,D), get_train_stats((A,B,C,D), l2_reg))
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Main Loop  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  2.1.0. width hyperparameter scan  -----------------------------

print("hey, let's train!")
nb_hiddens = 100
metrics = gradient_descend(nb_hiddens    = nb_hiddens ,
                           learning_rate = 0.01    ,
                           nb_steps      = 250+1      ,
                           report_every  =  10       , 
                           )

