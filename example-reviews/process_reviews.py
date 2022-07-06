'''
    EXPORTS
    lines
    shared_words, sw_by_idx, idx_by_sw
    feat
'''

LOWER_ALL = True
COUNT_THRESH = 3
MIN_WORD_COUNT = 3 
UNK = 'UNK'

okay = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz')
clean = (lambda s: ''.join(c for c in s if c in okay).lower()  if LOWER_ALL else
         lambda s: ''.join(c for c in s if c in okay)        )

def get_lines(file_nm='reviews_train.tsv'):
    with open(file_nm, encoding='latin1') as f:
        lines = [l.strip() for l in f.read().split('\n')]
        lines = [l.split('\t') for l in lines if l][1:]
        lines = [(int(l[0]), clean(l[3])) for l in lines if l]
        lines = [(s,r) for s,r in lines if MIN_WORD_COUNT <= len(r.split())]
    return lines

def get_shared_words(lines): 
    words = set(w for _,r in lines for w in r.split())
    counts = {w:0 for w in words}
    for _,r in lines:
        for w in r.split():
            counts[w] += 1
    shared_words = set(w for w in words if COUNT_THRESH<=counts[w]) 
    shared_words.add(UNK)
    sw_by_idx = sorted(shared_words) 
    idx_by_sw = {sw:i for i,sw in enumerate(sw_by_idx)}
    return shared_words, sw_by_idx, idx_by_sw

def featurize(review, shared_words):
    sws = [(w if w in shared_words else UNK) for w in review.split()]
    return [idx_by_sw[sw] for sw in sws] 

if __name__=='__main__':
    lines = get_lines('reviews_train.tsv')
    shared_words, sw_by_idx, idx_by_sw = get_shared_words(lines) 
    feat = lambda r: featurize(r, shared_words)

    N = len(lines)
    D = len(sw_by_idx)
    
    print('{} revs  '
          #'{:.1f} chars/rev  '
          '{:.1f}% pos  '
          '{} shared words  '.format(
        N,
        #float(sum(len(r) for _,r in lines)/N),
        100*float(sum(max(0,s) for s,_ in lines)/N),
        D,
        ))
    
    for s,r in lines[:20]:
        print(s, featurize(r, shared_words))

# RNN
