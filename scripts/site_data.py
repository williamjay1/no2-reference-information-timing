"""Frozen-period site fleets; initialization windows are external setup costs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from study_data import read_data, deployment_arrays, eligible, ROOT
from calibration_engine import initialize,guard_features


def make_site(site, stage='test', q_scale=1e-3, use_ar=True,
              weather=True, ar_share=.8,guard_width=8.):
    data = read_data()
    data = data[data.site.eq(site)]
    cutoff = pd.Timestamp('2021-01-01')
    if stage=='development':
        data=data[data.timestamp<cutoff]
    else:
        data=data[data.timestamp>=cutoff]
    # Ending a replay is an evaluation-design choice, not future availability
    # passed to the acquisition controller. Use the last ratified sensor pair.
    paired=data[np.isfinite(data.x)&np.isfinite(data.y)]
    start=data.timestamp.min()
    end=paired.timestamp.max()
    times=pd.date_range(start,end,freq='h')
    T=len(times)
    models=[]; ready=[]; metadata=[]; scores=[]; Xs=[]
    initmask=np.zeros(T,dtype=bool)
    reference=pd.read_parquet(ROOT/'datasets/reference_no2_ratified.parquet')
    reference=reference[reference.site.eq(site)]
    y=reference.set_index('timestamp').y.reindex(times).to_numpy(float)
    for key,g in data.groupby('deployment',sort=True):
        local=deployment_arrays(g,stop=end,weather=weather)
        ok,score=eligible(local)
        if not ok:
            continue
        k=int((local['times'][0]-start).total_seconds()/3600)
        n=len(local['times'])
        guarded,guard=guard_features(local['X'],width=guard_width)
        model=initialize(guarded,local['y'],336,q_scale=q_scale,
                         use_ar=use_ar,ar_share=ar_share)
        phi=np.zeros((T,model['phi'].shape[1]))
        phi[k:k+n]=model['phi']
        valid=np.zeros(T,dtype=bool); valid[k:k+n]=model['valid']
        model['phi']=phi; model['valid']=valid
        sc=np.zeros(T,dtype=bool); sc[k:k+n]=score
        xx=np.full((T,local['X'].shape[1]),np.nan); xx[k:k+n]=local['X']
        models.append(model); ready.append(k+336); scores.append(sc); Xs.append(xx)
        initmask[k:min(k+336,T)]=True
        row=dict(site=site,deployment=key,device=local['device'],
                 instrument=g.instrument.iloc[0],family=local['family'],
                 start=str(local['times'][0]),end=str(local['times'][-1]),
                 ready=str(times[min(k+336,T-1)]),init_pairs=model['init_pairs'],
                 scored=int(score.sum()),rho=model['rho'],local_offset=k)
        row['guard']=guard
        row['guarded_eval_hours']=int(np.sum(score & (guarded[:,0]!=local['X'][:,0])))
        metadata.append(row)
    # Concurrent channels divide one instrument's acquisition weight. Two
    # replacement segments do not overlap and therefore retain unit weight.
    for i,row in enumerate(metadata):
        concurrent=max(1,sum((r['instrument']==row['instrument']) and
                            (r['start']==row['start']) and (r['end']==row['end'])
                            for r in metadata))
        models[i]['weight']=1./concurrent
    return dict(models=models,y=y,ready=np.asarray(ready,dtype=np.int64),
                initialization_mask=initmask,metadata=metadata,
                score_masks=np.asarray(scores),Xs=Xs,times=times,
                stage=stage,site=site)


if __name__=='__main__':
    summary=[]
    for s in ['Manchester','London','York']:
        d=make_site(s)
        summary.extend(d['metadata'])
        print(s,len(d['models']),len(d['times']),sum(r['scored'] for r in d['metadata']))
    (ROOT/'results/test_eligibility.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
