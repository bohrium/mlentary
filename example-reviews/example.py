''' author: sam tenka
    change: 2022-06-29
    create: 2022-06-13
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
    to use: Run `python3 example.py`.  On first run, expect a downloading
            progress bar to display and finish within 30 to 60 seconds; this
            is for downloading the MNIST dataset we'll use.
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
from process_reviews import get_lines, get_shared_words, featurize

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

DATA_DIM     = None

NB_TEST      = 3000
NB_TRAIN_MAX = 1000
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

#--------------  0.2.1. load reviews data  -------------------------------------

train_lines = get_lines('reviews_train.tsv')
shared_words, _, _  = get_shared_words(train_lines) 
feat = lambda r: featurize(r, shared_words)
 
test_lines = get_lines('reviews_test.tsv')

#--------------  0.2.2. format data  ------------------------------------------- 



#--------------  0.2.3. shuffle and split  -------------------------------------

idxs = np.arange(len(all_y))
np.random.shuffle(idxs)
all_x = all_x[idxs]
all_y = all_y[idxs]
all_y_sign = np.array([+1 if y==1 else -1 for y in all_y]) 
#
train_idxs = np.arange(0           , NB_TRAIN            )
test_idxs  = np.arange(NB_TRAIN_MAX, NB_TRAIN_MAX+NB_TEST)

#===============================================================================
#==  1. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.0. Define Linear Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if BIAS:
    lrelu = lambda z: np.maximum(z*LEAK, z)
    dlrelu = lambda z: 1 + (LEAK-1) * (1.0-np.sign(z))/2.0
else:
    lrelu = lambda z: np.concatenate((np.maximum(z*LEAK, z), [1.0]))
    dlrelu = lambda z: np.concatenate((1 + (LEAK-1) * (1.0-np.sign(z))/2.0, [1.0]))

#make_dec_func_1 = lambda A: lambda x: np.dot(A, x)
make_dec_func_2 = lambda A,B: lambda x: np.dot(A, lrelu(np.matmul(B,x)))
#make_dec_func_3 = lambda A,B,C: lambda x: np.dot(A, lrelu(np.matmul(B, lrelu(np.matmul(C,x)))))
make_dec_func   = make_dec_func_2

make_classifier = lambda dec_func: lambda x: 0 if dec_func(x)<=0 else 1

is_correct = lambda classifier, idx: 1 if all_y[idx]==classifier(all_x[idx]) else 0

error_rate = lambda classifier, idxs: np.mean([1.0-is_correct(classifier, idx)
                                            for idx in idxs])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.1. Train and Test Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.1.0. loss functions and gradients  --------------------------

hinge     = lambda z: 0 if 1+z<0 else 1+z 
dhinge    = lambda z: 0 if 1+z<0 else   1 

#softplus  = lambda z: np.log(1. + np.exp(z))
#dsoftplus = lambda z: 0 if 1+z<0 else   1 

def gradient(weights, y_sign,x, l2_reg): 
    A,B = weights 
    # forward prop:
    z = np.matmul(B,x)
    h = lrelu(z) 
    o = np.dot(A, h) 
    l = hinge(-y_sign*o)
    # back prop:
    dl_do = -y_sign*dhinge(-y_sign*o)
    dl_dh = A * dl_do 
    dl_dz = dlrelu(z) * dl_dh
    return (dl_do * h          + l2_reg*A,
            np.outer(dl_dz, x) + l2_reg*B, l)

def gradients(weights, idxs=train_idxs, l2_reg=0.01): 
    gs = [gradient(weights,all_y_sign[i],all_x[i], l2_reg) for i in idxs]
    gAs = [g[0] for g in gs] 
    gBs = [g[1] for g in gs] 
    ls  = [g[2] for g in gs] 
    return (np.mean(gAs, axis=0),
            np.mean(gBs, axis=0), np.mean(ls))

#--------------  1.1.2. gradient descent  --------------------------------------

def get_train_stats(weights, l2_reg):
    dec_func = make_dec_func(*weights)
    clsfier = make_clsfier(dec_func)
    _, _, l = gradients(weights, train_idxs, l2_reg)
    return {'train-loss':l,
            'train-err':error_rate(clsfier, train_idxs),
            'test-err' :error_rate(clsfier, test_idxs ), 
            }

def gradient_descend(nb_hiddens, learning_rate,
                     nb_steps, report_every=None, batch_size=32, l2_reg=0.01,
                     weights = None):
    # initialize:
    A,B = weights if weights is not None else (
        np.random.randn(nb_hiddens+1),
        np.random.randn(nb_hiddens+1,DATA_DIM+1)/np.sqrt(nb_hiddens+DATA_DIM)
    )
    print('training with '
            '\033[33m{:2d}\033[34m hiddens and '
            'learning rate \033[33m{:.4f}\033[34m...'.format(nb_hiddens, learning_rate))

    # main loop:
    for t in range(nb_steps):
        # sgd update: 
        batch = np.random.choice(train_idxs, batch_size, replace=False)
        gA, gB, l = gradients((A,B), batch, l2_reg)
        A -= 1.0 * learning_rate * gA * np.random.choice(2, gA.shape) 
        B -= 1.0 * learning_rate * gB * np.random.choice(2, gB.shape)
         
        # report progress during training loop:
        if report_every is None or t%report_every: continue

        # print numeric scores:
        train_stats = get_train_stats((A,B), l2_reg)
        print('     after \033[33m{:6d}\033[34m steps: '
              'train-loss=\033[33m{:6.3f}\033[34m '
               'train-err=\033[33m{:5.1f}%\033[34m '
                'test-err=\033[33m{:5.1f}%\033[34m '.format(
            t,
            train_stats['train-loss'],
            100*train_stats['train-err'],
            100*train_stats['test-err'],
            ))

    ## plot decision boundary etc
    #print('  plotting...'.format(nb_hiddens))
    #plot_features(
    #    idxs=train_idxs,
    #    dec_func = make_dec_func(A,B),
    #    interesting_params = B,
    #    ip_weights = A,
    #    file_name='reviews-{:03d}-{:05d}.png'.format(nb_hiddens,t),
    #    opacity_factor=0.75,

    return ((A,B), get_train_stats(weights, l2_reg))
   
##===============================================================================
##==  2. PLOT  ==================================================================
##===============================================================================
#
##--------------  2.0.1. define scatter plot initializer  -----------------------
#
#def new_plot(data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
#             nb_vert_axis_ticks=10, nb_hori_axis_ticks=10): 
#    # white canvas 
#    scatter = np.ones((data_h+2*margin,
#                       data_w+2*margin,3), dtype=np.float32) 
#
#    # grid lines
#    for a in range(nb_vert_axis_ticks): 
#        s = int(data_h * float(a)/nb_vert_axis_ticks)
#        scatter[margin+(data_h-1-s),margin:margin+data_w] = SMOKE
#    for a in range(nb_hori_axis_ticks): 
#        s = int(data_w * float(a)/nb_hori_axis_ticks)
#        scatter[margin:margin+data_h,margin+s]            = SMOKE
#    
#    # tick marks
#    for a in range(nb_vert_axis_ticks): 
#        s = int(data_h * float(a)/nb_vert_axis_ticks)
#        for i in range(nb_vert_axis_ticks)[::-1]:
#            color = SLATE + 0.04*i*WHITE
#            scatter[margin+(data_h-1-s)               ,  :margin+2+i] = color
#    for a in range(nb_hori_axis_ticks): 
#        s = int(data_w * float(a)/nb_hori_axis_ticks)
#        for i in range(nb_hori_axis_ticks)[::-1]:
#            color = SLATE + 0.04*i*WHITE
#            scatter[margin+data_h-2-i:2*margin+data_h , margin+s    ] = color
#   
#    # axes
#    scatter[margin+data_h-1      , margin:margin+data_w] = SLATE
#    scatter[margin:margin+data_h , margin              ] = SLATE
#
#    return scatter
#
##--------------  2.0.2. define feature space scatter plot  --------------------
#
#def plot_features(idxs=train_idxs,file_name='new-train.png', opacity_factor=1.0,
#                  min_vert=-1.0, max_vert=1.0,  min_hori=-1.0, max_hori=1.0,
#                  interesting_params=[], ip_weights=None, dec_func_maker=None,
#                  data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
#                  nb_vert_axis_ticks=10, nb_hori_axis_ticks=10, dec_func=None):
#
#    # initialize
#    scatter = new_plot(data_h, data_w, margin,
#                       nb_vert_axis_ticks, nb_hori_axis_ticks)
#
#    # color in decision boundary 
#    if dec_func is not None:
#        for r in range(data_h):
#            for c in range(data_w):
#                z0 = BIAS 
#                z1 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
#                z2 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))
#
#                dec = dec_func((z0,z1,z2)) 
#                if abs(dec) > 0.1 : continue
#
#                signs = lambda thresh:len(set([
#                        np.sign(dec_func((z0,z1+thresh*i,z2+thresh*j)))
#                        for i in range(-1,2) for j in range(-1,2)      ]))
#
#                opacity = np.mean([(1.0 if signs(thresh)==2 else 0.0) for thresh
#                        in (0.002,0.004,0.006)])
#                if opacity==0.0: continue
#
#                rr = margin+r
#                cc = margin+c
#                overlay_color(scatter[rr-1:rr+2,cc       ], SLATE, opacity*0.10)
#                overlay_color(scatter[rr-2:rr+3,cc       ], SMOKE, opacity*0.02)
#                overlay_color(scatter[rr       ,cc-1:cc+2], SLATE, opacity*0.10)
#                overlay_color(scatter[rr       ,cc-2:cc+3], SMOKE, opacity*0.02)
#
#                overlay_color(scatter[rr       ,cc       ], SHADE, opacity*0.50)
#
#    # color in each hypothesis
#    for i,(p0,p1,p2) in list(enumerate(interesting_params)):
#        for r in range(data_h):
#            for c in range(data_w):
#                z0 = BIAS 
#                z1 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
#                z2 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))
#                rr = np.sqrt(p0**2+p1**2+p2**2)
#                q0,q1,q2 = p0/rr, p1/rr, p2/rr
#                dec = q0*z0+q1*z1+q2*z2
#
#                if dec<=-0.02: continue
#                opa = (0.05 * np.exp(-250.0*abs(dec)) if dec<0 else
#                       0.05 + 0.25 * min(1,rr*abs(ip_weights[i])*(dec/2.5)))
#
#                color = SHADE if ip_weights is None else (
#                        SHADE + min(1,2.0*rr*abs(ip_weights[i]))*((CYAN if ip_weights[i]<0 else RED)-SHADE) 
#                        )
#
#                overlay_color(scatter[margin+r,margin+c], color, opa)
#
#    # color in data scatter
#    for idx in idxs:
#        r = margin+data_h-1-int((data_h-1) * min(1.,max(0.,(all_x[idx][1]-min_vert)/(max_vert-min_vert))))
#        c = margin+         int((data_w-1) * min(1.,max(0.,(all_x[idx][2]-min_hori)/(max_hori-min_hori))))
#        color = CYAN if all_y[idx]==0 else RED
#        for dr in range(-margin,margin+1):
#            for dc in range(-margin,margin+1):
#                opa = opacity_factor * (2.0/float(2.0 + dr*dr+dc*dc))**2
#                overlay_color(scatter[r+dr,c+dc], color, opa)
#    
#    # save
#    plt.imsave(file_name, scatter) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Main Loop  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  2.1.0. width hyperparameter scan  -----------------------------

print("hey, let's train!")
for nb_hiddens in [1,2,3,5,8,13,21,34,55]:
    metrics = gradient_descend(nb_hiddens    = nb_hiddens   ,
                               learning_rate = 0.20         ,
                               nb_steps      = 10000+1      ,
                               report_every  =  1000        , 
                              )

