"""Independent comparison against raw MAT fields, not the extraction helpers."""
from pathlib import Path
import json
import h5py, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
cells=pd.read_csv(ROOT/'data/processed/cell_features.csv').set_index('cell_id')
summary=pd.read_csv(ROOT/'data/processed/cycle_summary.csv')
manifest=json.loads((ROOT/'data/processed/data_manifest.json').read_text())
counts={'raw_cells_compared':0,'summary_fields_compared':0,'raw_curves_compared':0,'early_features_compared':0}
arrays={name:np.load(ROOT/f'data/processed/{name}_curves.npz') for name in ['q10','q100','delta','voltage']}
def vec(f,d):
 a=d[()]
 if h5py.check_dtype(ref=a.dtype):a=np.concatenate([f[r][()].ravel() for r in a.ravel()])
 return np.asarray(a,dtype=float).ravel()
for m in manifest:
 with h5py.File(ROOT/'data/raw'/m['file'],'r') as f:
  batch=f['batch'];ids=[f'b{m["batch"][-1]}c{i}' for i in range(m['raw_cells'])]
  for i,cid in enumerate(ids):
   row=cells.loc[cid];sg=f[batch['summary'][i,0]];cg=f[batch['cycles'][i,0]]
   rawlife=vec(f,f[batch['cycle_life'][i,0]])[0];np.testing.assert_allclose(row.cycle_life,rawlife,equal_nan=True)
   policy=''.join(map(chr,f[batch['policy_readable'][i,0]][()].ravel().astype(int))).strip('\x00')
   assert row.charging_policy==policy
   cyc=vec(f,sg['cycle']);df=summary[summary.cell_id==cid];np.testing.assert_array_equal(df.cycle,cyc)
   assert len(cyc)==cg['Qdlin'].size and np.all(np.diff(cyc)>0)
   for dest,src in {'QD':'QDischarge','QC':'QCharge','IR':'IR','Tavg':'Tavg','Tmax':'Tmax','Tmin':'Tmin','chargetime':'chargetime'}.items():
    a=vec(f,sg[src]);np.testing.assert_allclose(df[dest],a,equal_nan=True,rtol=1e-12,atol=1e-12);counts['summary_fields_compared']+=1
    clean=a.copy();bad=~np.isfinite(a)
    if dest in ['QD','QC']:bad|=(a<=0)|(a>1.32)
    elif dest=='IR':bad|=a<=0
    elif dest in ['Tavg','Tmax','Tmin']:bad|=(a<=0)|(a>100)
    else:bad|=a<=0
    clean[bad]=np.nan;early=clean[(cyc>=1)&(cyc<=100)];early=early[np.isfinite(early)]
    np.testing.assert_allclose(row['mean_'+dest.lower()],early.mean() if len(early) else np.nan,equal_nan=True)
    assert row['invalid_'+dest]==bad.sum();counts['early_features_compared']+=1
    if dest=='QD':
     mask=(cyc>=10)&(cyc<=100)&np.isfinite(clean)
     x=cyc[mask];y=clean[mask]
     slope=((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum()
     np.testing.assert_allclose(row.qd_slope_10_100,slope,rtol=1e-10,atol=1e-12)
     assert row.eol_not_observed==(clean[np.isfinite(clean)][-1]>.885)
   for cycle,key in [(10,'q10'),(100,'q100')]:
    j=int(np.where(cyc==cycle)[0][0]);raw=vec(f,f[cg['Qdlin'][()].ravel()[j]])
    np.testing.assert_allclose(arrays[key][cid],raw,equal_nan=True);counts['raw_curves_compared']+=1
   np.testing.assert_array_equal(arrays['voltage'][cid],vec(f,f[batch['Vdlin'][i,0]]))
   delta=arrays['q100'][cid]-arrays['q10'][cid]
   np.testing.assert_allclose(arrays['delta'][cid],delta,equal_nan=True)
   np.testing.assert_allclose(row.log_var_delta_q,np.log10(delta.var(ddof=1)),rtol=1e-12)
   counts['raw_cells_compared']+=1
corr=pd.read_csv(ROOT/'results/feature_correlations.csv')
for _,r in corr.iterrows():
 pair=cells[cells.batch==r.batch][[r.feature,'cycle_life']].dropna()
 assert r.n==len(pair)
 np.testing.assert_allclose(r.spearman,spearmanr(pair.iloc[:,0],pair.iloc[:,1]).statistic,atol=1e-12)
for a in arrays.values():a.close()
counts.update(correlations_compared=len(corr),passed=True)
(ROOT/'results/independent_review.json').write_text(json.dumps(counts,indent=2))
print(counts)
