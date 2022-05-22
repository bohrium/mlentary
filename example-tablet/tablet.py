import numpy as np 
from matplotlib import pyplot as plt
#import matplotlib as mpl
#import random # DON'T USE (since want np.random seed, not random seed!)

np.random.seed(419)

NB_DOCS = 120
DOC_LEN_IN_WORDS = 10
NB_SYLLABLES = 120
NB_SUB_RULES_PER_TYPE = 120
NB_WORDS = 640

from tablet_data import MEANS, MEAN_DIM, GRAMS, GRAM_DIM 

def sample(L, ps):
    r = np.random.random()
    for l,p in zip(L,ps):
        r-=p
        if r<=0 : return l
def pair_with_probs(L, len_power=0, sort=False):
    freqs = np.random.exponential(size=len(L)) 
    if sort:
        freqs = np.array(sorted(freqs, reverse=True))
    if len_power!=0:
        freqs *= np.array([len(l) for l in L])**len_power
    freqs /= np.sum(freqs)
    return L,freqs

vowels, vfreqs = pair_with_probs('eaoiu'                , sort=True) #'aeiou')
consns, cfreqs = pair_with_probs('tnshdrlcmwfgypbvkjxqz', sort=True) #'bcdfghjklmnpqrstvwxyz')

syllables = set([(C0+V+C1) for _ in range(NB_SYLLABLES)
                           for C0 in [sample(consns, cfreqs)]
                           for V  in [sample(vowels, vfreqs)]
                           for C1 in [sample(consns, cfreqs)]])

syllables = sorted(syllables)

print('{} syllables'.format(len(syllables)))
for K in range(0,len(syllables),12):
    print('  '.join(syllables[K:K+12]))
print()

meanings_repr = np.random.randn(len(syllables), MEAN_DIM)
meanings_repr /= np.expand_dims(np.linalg.norm(meanings_repr, axis=1), axis=1)
function_repr = np.random.randn(len(syllables), GRAM_DIM)
function_repr /= np.expand_dims(np.linalg.norm(function_repr, axis=1), axis=1)

syllables, sfreqs = pair_with_probs(syllables)

vcc_sub_rules = set([(V+C0+C1,V  ) for _ in range(NB_SUB_RULES_PER_TYPE)
                                   for V  in [sample(vowels, vfreqs)] 
                                   for C0 in [sample(consns, cfreqs)]
                                   for C1 in [sample(consns, cfreqs)]
                                   for C  in [sample(consns, cfreqs)]])
ccv_sub_rules = set([(C0+C1+V,  C) for _ in range(NB_SUB_RULES_PER_TYPE)
                                   for V  in [sample(vowels, vfreqs)] 
                                   for C0 in [sample(consns, cfreqs)]
                                   for C1 in [sample(consns, cfreqs)]
                                   for C  in [sample(consns, cfreqs)]])
vvc_sub_rules = set([(V0+V1+C,  V) for _ in range(NB_SUB_RULES_PER_TYPE)
                                   for C  in [sample(consns, cfreqs)] 
                                   for V0 in [sample(vowels, vfreqs)]
                                   for V1 in [sample(vowels, vfreqs)]
                                   for V  in [sample(vowels, vfreqs)]])
cvc_sub_rules = set([(C0+V+C1,V+C) for _ in range(NB_SUB_RULES_PER_TYPE)
                                   for V  in [sample(vowels, vfreqs)] 
                                   for C0 in [sample(consns, cfreqs)]
                                   for C1 in [sample(consns, cfreqs)]
                                   for C  in [sample(consns, cfreqs)]])

sub_rules = (sorted(vcc_sub_rules)
            +sorted(ccv_sub_rules)
            +sorted(vvc_sub_rules)
            +sorted(cvc_sub_rules))

def triplet():
    return tuple(sample(range(len(syllables)), sfreqs) for _ in range(3))

def spelling_of(triplet):
    base = ''.join(syllables[s] for s in triplet)
    for (k,v) in sub_rules:
        base = base.replace(k,v)
    return base

etyms = {spelling_of(t):t for _ in range(NB_WORDS)
                          for t in [triplet()]  }
words = sorted(etyms.keys())

def affinity(trip, mean_a_idx, mean_b_idx, gram_idx):
    return min(sum(meanings_repr[s_idx][mean_a_idx] for s_idx in trip[-3:]),
               sum(meanings_repr[s_idx][mean_b_idx] for s_idx in trip[-2:]),
               sum(function_repr[s_idx][gram_idx  ] for s_idx in trip[-1:])) 

def lookup(mean_a, mean_b, gram):
    mean_a_idx = MEANS.index(mean_a)
    mean_b_idx = MEANS.index(mean_b)
    gram_idx   = GRAMS.index(gram  )
    return max((affinity(trip, mean_a_idx, mean_b_idx, gram_idx), k)
               for k,trip in etyms.items())[1]

print('{} words'.format(len(words)))
for K in range(0,len(words),18):
    print('  '.join(words[K:K+18]))
print()

words, wfreqs = pair_with_probs(words, len_power=-2.0)
for _ in range(20):
    print(' '.join(sample(words, wfreqs) for _ in range(10)))
print()

#print(lookup('cow'   , 'cow', 'verb'))
#print(lookup('cow'   , 'cow', 'noun'))
#print(lookup('cow'   , 'cow', 'adj' ))
#print(lookup('cow'   , 'cow', 'prep'))
#print()
#
#print(lookup('after' , 'love' , 'verb'))
#print(lookup('before', 'love' , 'verb'))
#print(lookup('hate'  , 'love' , 'verb'))
#print(lookup('love'  , 'love' , 'verb'))
#print()
#
#print(lookup('farm'  , 'tool' , 'noun'))
#print(lookup('war'   , 'tool' , 'noun'))
#print(lookup('king'  , 'tool' , 'noun'))
#print(lookup('horse' , 'tool' , 'noun'))
#print()
#
#print(lookup('after' , 'soul' , 'noun'))
#print(lookup('before', 'soul' , 'noun'))
#print(lookup('after' , 'soul' , 'verb'))
#print(lookup('before', 'soul' , 'verb'))
#print()
