"""Prepare initialization-scaled binary streams for ReFIT.

Extracted from the original runner dependency, with configurable input paths.
The original VERSION string is retained for existing cache compatibility.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
VERSION='refit-prequential-query-only-v1'

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(p)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(dataset,cache,data_root=None):
    if dataset not in {"2017", "2023", "2025"}:
        raise ValueError("dataset must be 2017, 2023, or 2025")
    cache=Path(cache)
    data_root=Path(data_root) if data_root is not None else Path("data")
    folder=cache/dataset;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        for name in ['X.npy','y.npy','scaler.npz']:
            if not (folder/name).is_file():
                raise FileNotFoundError(f'Incomplete cache: {folder/name}; rebuild in a new cache directory')
        return folder
    source=[];dropped=0;init_indices=None
    if dataset=='2017':
        source=[data_root/'2017'/f'CIC_17_day{i}.csv' for i in range(1,6)]
        xx=[];yy=[];n=0;features=None
        for file in source:
            for frame in pd.read_csv(file,chunksize=100000):
                lc=next(c for c in frame if c.strip().lower()=='label')
                if features is None:features=[c for c in frame if c!=lc and not c.startswith('Unnamed')]
                vals=frame[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype='float64')
                ok=np.isfinite(vals).all(1)&frame[lc].notna().to_numpy();dropped+=int((~ok).sum())
                x=vals[ok];y=(frame.loc[ok,lc].astype(str).str.strip().str.lower()!='benign').to_numpy(dtype='int64')
                remain=2800000-n;xx.append(x[:remain]);yy.append(y[:remain]);n+=len(xx[-1])
                if n>=2800000:break
            if n>=2800000:break
        X=np.concatenate(xx);Y=np.concatenate(yy);del xx,yy
        if len(Y)!=2800000 or Y[:500000].sum()!=0:raise ValueError('2017 stream does not satisfy published initialization/length')
        bounds=[(0,500000)]+[(i,i+50000) for i in range(500000,2800000,50000)]
    elif dataset=='2023':
        base=data_root/'2023'
        source=[base/'train.csv',base/'test.csv'];train=pd.read_csv(source[0]);test=pd.read_csv(source[1])
        features=[c for c in train if c not in ['y_binary','y_class','y_family'] and not c.startswith('Unnamed')]
        # Fixed stratified predeployment split, preserving order within source.
        init_indices=np.sort(np.concatenate([np.flatnonzero(train.y_binary.to_numpy()==k)[:25000] for k in [0,1]]))
        if len(init_indices)!=50000:raise ValueError('Insufficient initialization class counts')
        df=pd.concat([train.iloc[init_indices],test],ignore_index=True)
        X=df[features].to_numpy(dtype='float64');Y=df.y_binary.to_numpy(dtype='int64')
        if len(Y)!=1850000 or not np.isfinite(X).all():raise ValueError('2023 stream integrity mismatch')
        bounds=[(i,i+50000) for i in range(0,len(Y),50000)]
    else:
        source=[data_root/'2025'/'Main_CIC_IIoT2025_forced_drift_binary_light_ref_scaled.csv']
        df=pd.read_csv(source[0]);features=[c for c in df if c not in ['window_id','label']]
        X=df[features].to_numpy(dtype='float64');Y=df.label.to_numpy(dtype='int64')
        ids=df.window_id.to_numpy();splits=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
        bounds=list(zip(splits[:-1].tolist(),splits[1:].tolist()))
        if len(bounds)!=23 or len(Y)!=227191 or Y[:10000].sum()!=500:raise ValueError('2025 stream does not match expected source windows')
        if not np.isfinite(X).all():raise ValueError('Nonfinite 2025 features')
    assert set(np.unique(Y))<={0,1}
    n0=bounds[0][1];scaler=StandardScaler().fit(X[:n0]);X=scaler.transform(X).astype('float32')
    if not np.isfinite(X).all():raise ValueError('Scaling produced nonfinite features')
    np.save(folder/'X.npy',X);np.save(folder/'y.npy',Y)
    np.savez(folder/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    if init_indices is not None:np.save(folder/'initialization_source_indices.npy',init_indices)
    windows=[dict(window=i,start=a,end=b,n=b-a,benign=int((Y[a:b]==0).sum()),attack=int(Y[a:b].sum())) for i,(a,b) in enumerate(bounds)]
    dump(folder/'manifest.json',dict(version=VERSION,dataset=dataset,features=features,windows=windows,dropped_invalid=dropped,
         preprocessing='StandardScaler fitted on initialization window only; fixed feature order',
         initialization_selection='first 25000 per class from predeployment train pool, then source order; pool has 200000 pre-labeled rows' if dataset=='2023' else 'first source window(s)',
         sources=[dict(path=str(f),sha256=digest(f),bytes=f.stat().st_size) for f in source],
         cache_sha256={n:digest(folder/n) for n in ['X.npy','y.npy','scaler.npz']}))
    return folder


def from_counts(tp,fp,tn,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.;f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=p,recall=r,f1=f)
