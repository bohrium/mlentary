''' author: sam tenka
    change: 2022-07-05
    create: 2022-06-13
    descrp: 
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
                clsfier     --- classifier
    thanks: 
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

NB_TEST      =  900
NB_TRAIN     =   50
#NB_TRAIN     =  100
NB_TRAIN_MAX = 5000
BIAS = 1.0

#--------------  0.1.1. plotting and writing parameters  -----------------------

PLT_SIDE = 150
MARG     = 2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.2. global initialization and computations  ~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.2.0. parameterize randomness for replicability  -------------

np.random.seed(0)

#--------------  0.2.1. define artificial galaxy data  -------------------------

sym  = lambda: 2.15*(np.random.random()-0.5)
sym3 = lambda: sym()*sym()*sym()
choice = lambda l: l[np.random.choice(len(l))]

make_neg = lambda: choice([
    (1.0*sym3(), 1.0*sym ()), 
    (1.0*sym (), 1.0*sym3()), 
    ])
make_pos = lambda: choice([
    (+0.6+2.*0.2*sym3(), +0.0+2.*0.8*sym3()),
    (-0.3+2.*0.5*sym3(), +0.3+2.*0.5*sym3()),
    (-0.3+2.*0.8*sym3(), -0.3+2.*0.2*sym3()),
    ])

clip = lambda x: min(1, max(-1, x))
def post_process(p,f=3.0,o=0.1):
    p0,p1 = p
    p0 -= o 
    r = np.sqrt(p0**2+p1**2)
    z0, z1 = ( p0*np.cos(f*r) + p1*np.sin(f*r) + o,
              -p0*np.sin(f*r) + p1*np.cos(f*r)       )
    return (BIAS, clip((0.0+z0)/2.0), 
                  clip((0.0+z1)/2.0))

#--------------  0.2.2. generate artificial galaxy data  -----------------------

all_y = np.array([np.random.randint(2) for _ in range(NB_TRAIN_MAX+NB_TEST)])
all_x = np.array([post_process((make_neg, make_pos)[y]())
                  for y in all_y]) 

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
#==  2. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.0. Define Linear Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lrelu = lambda z: np.maximum(z*0.1, z)
dlrelu = lambda z: 1 + (0.1-1) * (1.0-np.sign(z))/2.0

make_dec_func = lambda A,B: lambda x: np.dot(A, lrelu(np.matmul(B,x)))
make_dec_func_= lambda A,B,C: lambda x: np.dot(A, lrelu(np.matmul(B,lrelu(np.matmul(C,x)))))
make_clsfier = lambda dec_func: lambda x: 0 if dec_func(x)<=0 else 1

is_correct = lambda clsfier, idx: 1 if all_y[idx]==clsfier(all_x[idx]) else 0

error_rate = lambda clsfier, idxs: np.mean([1.0-is_correct(clsfier, idx)
                                            for idx in idxs])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Train and Test Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  2.1.0. loss functions and gradients  --------------------------

hinge  = lambda z: 0 if 1+z<0 else 1+z 
dhinge = lambda z: 0 if 1+z<0 else   1 

def gradient(A,B, y_sign,x, l2_reg=0.0): 
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

def gradients(A,B, idxs=train_idxs, l2_reg=0.0): 
    gs = [gradient(A,B,all_y_sign[i],all_x[i], l2_reg) for i in idxs]
    gAs = [g[0] for g in gs] 
    gBs = [g[1] for g in gs] 
    ls  = [g[2] for g in gs] 
    return (np.mean(gAs, axis=0),
            np.mean(gBs, axis=0), np.mean(ls))

#--------------  2.1.2. gradient descent  --------------------------------------

#===============================================================================
#==  3. PLOT  ==================================================================
#===============================================================================

#--------------  3.0.1. define scatter plot initializer  -----------------------

def new_plot(data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
             nb_vert_axis_ticks=10, nb_hori_axis_ticks=10): 
    # white canvas 
    scatter = np.ones((data_h+2*margin,
                       data_w +2*margin,3), dtype=np.float32) 

    # grid lines
    for a in range(nb_vert_axis_ticks): 
        s = int(data_h * float(a)/nb_vert_axis_ticks)
        scatter[margin+(data_h-1-s),margin:margin+data_w] = SMOKE
    for a in range(nb_hori_axis_ticks): 
        s = int(data_w * float(a)/nb_hori_axis_ticks)
        scatter[margin:margin+data_h,margin+s]            = SMOKE
    
    # tick marks
    for a in range(nb_vert_axis_ticks): 
        s = int(data_h * float(a)/nb_vert_axis_ticks)
        for i in range(nb_vert_axis_ticks)[::-1]:
            color = SLATE + 0.04*i*WHITE
            scatter[margin+(data_h-1-s)               ,  :margin+2+i] = color
    for a in range(nb_hori_axis_ticks): 
        s = int(data_w * float(a)/nb_hori_axis_ticks)
        for i in range(nb_hori_axis_ticks)[::-1]:
            color = SLATE + 0.04*i*WHITE
            scatter[margin+data_h-2-i:2*margin+data_h , margin+s    ] = color
   
    # axes
    scatter[margin+data_h-1      , margin:margin+data_w] = SLATE
    scatter[margin:margin+data_h , margin              ] = SLATE

    return scatter

#--------------  3.0.2. define feature space scatter plot  --------------------

def plot_features(file_name, idxs=train_idxs, opacity_factor=1.0,
                  min_vert=-1.0, max_vert=1.0,  min_hori=-1.0, max_hori=1.0,
                  data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
                  nb_vert_axis_ticks=10, nb_hori_axis_ticks=10, dec_func=None):

    # initialize
    scatter = new_plot(data_h, data_w, margin,
                       nb_vert_axis_ticks, nb_hori_axis_ticks)

    if dec_func is not None:
        for r in range(data_h):
            for c in range(data_w):
                z0 = BIAS 
                z1 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
                z2 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))

                dec = dec_func((z0,z1,z2)) 

                # general hueing:
                color = (SHADE + (CYAN-SHADE)*min(1.0, 0.2*-dec) if dec<0.0 else 
                         SHADE + (RED -SHADE)*min(1.0, 0.2*+dec)) 
                overlay_color(scatter[margin+r,margin+c], color, 0.2)

                if abs(dec) > 0.15: continue

                # color in decision boundary 
                signs = lambda thresh:len(set([
                        np.sign(dec_func((z0,z1+thresh*i,z2+thresh*j)))
                        for i in range(-1,2) for j in range(-1,2)      ]))

                opacity = np.mean([(1.0 if signs(thresh*(10.))==2 else 0.0) for thresh
                    in (0.002,0.004,0.006)])
                if opacity==0.0: continue

                rr = margin+r
                cc = margin+c
                overlay_color(scatter[rr-1:rr+2,cc       ], SLATE, opacity*0.10)
                overlay_color(scatter[rr-2:rr+3,cc       ], SMOKE, opacity*0.02)
                overlay_color(scatter[rr       ,cc-1:cc+2], SLATE, opacity*0.10)
                overlay_color(scatter[rr       ,cc-2:cc+3], SMOKE, opacity*0.02)
                overlay_color(scatter[rr       ,cc       ], SHADE, opacity*0.50)

    # save
    plt.imsave(file_name, scatter) 

def sketch2(nb_hiddens, idx=0):
    # initialize:
    A = 2*np.random.randn(nb_hiddens)/np.sqrt(nb_hiddens+1)
    B = 2*np.random.randn(nb_hiddens,3)/np.sqrt(nb_hiddens+3)
    plot_features(
        idxs=train_idxs,
        dec_func = make_dec_func(A,B),
        file_name='sketch-2-{:03d}-{:03d}.png'.format(nb_hiddens, idx),
        opacity_factor=0.75,
        ) 
def sketch3(nb_hiddens_out, nb_hiddens_in, idx=0):
    # initialize:
    A = 2*np.random.randn(nb_hiddens_out)/np.sqrt(nb_hiddens_out+1)
    B = 2*np.random.randn(nb_hiddens_out,nb_hiddens_in)/np.sqrt(nb_hiddens_out+nb_hiddens_in)
    C = 2*np.random.randn(nb_hiddens_in,3)/np.sqrt(nb_hiddens_in+3)
    plot_features(
        idxs=train_idxs,
        dec_func = make_dec_func_(A,B,C),
        file_name='sketch-3-{:03d}-{:03d}-{:03d}.png'.format(nb_hiddens_out, nb_hiddens_in, idx),
        min_vert=-5.0, max_vert=5.0,  min_hori=-5.0, max_hori=5.0,
        opacity_factor=0.75,
        ) 

#for idx in tqdm.tqdm(range(50)): 
#    for nb_hiddens in [64]:
#        sketch2(nb_hiddens, idx)

for idx in tqdm.tqdm(range(50)): 
    for nb_hiddens_out in [64]:
        for nb_hiddens_in in [ 4]:
            sketch3(nb_hiddens_out, nb_hiddens_in, idx)

#for nb_hiddens in tqdm.tqdm(range(4,20+1,4)):
#    sketch2(nb_hiddens)
#
#for nb_hiddens_out in tqdm.tqdm(range(4,20+1,4)):
#    for nb_hiddens_in in tqdm.tqdm(range(4,20+1,4)):
#        sketch3(nb_hiddens_out, nb_hiddens_in)
