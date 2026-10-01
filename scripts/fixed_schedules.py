"""Known initialization calendar is the only future scheduling information."""
import numpy as np


def fixed_site_mask(n,origin,initmask,budget=48,block=672,max_starts=4,
                    phase=.25,seed=None):
    rng=np.random.default_rng(seed)
    mask=np.zeros(n,dtype=bool)
    audit=[]
    for left in range(origin,n,block):
        right=min(left+block,n)
        target=min(int(budget*(right-left)//block),int((~initmask[left:right]).sum()))
        k=min(max_starts,target)
        if not k: continue
        lengths=np.full(k,target//k,dtype=int); lengths[:target%k]+=1
        for j,L in enumerate(lengths):
            if seed is None:
                preferred=left+int((j+phase)*(right-left)/k)
            else:
                preferred=int(rng.integers(left,right))
            candidates=[]
            for t in range(left,right-L+1):
                if np.any(initmask[t:t+L]) or np.any(mask[t:t+L]): continue
                # Keep episodes separated; one gap is enough.
                if (t>left and mask[t-1]) or (t+L<right and mask[t+L]): continue
                candidates.append(t)
            if not candidates:
                raise RuntimeError('Initialization calendar cannot accommodate fixed windows')
            t=min(candidates,key=lambda a:(abs(a-preferred),a))
            mask[t:t+L]=True
        actual=int(mask[left:right].sum())
        starts=int(np.sum(mask[left:right]&~np.r_[False,mask[left:right-1]]))
        if actual!=target or starts>max_starts: raise RuntimeError('Fixed budget/cap mismatch')
        audit.append(dict(left=left,right=right,target=target,actual=actual,starts=starts))
    return mask,audit
