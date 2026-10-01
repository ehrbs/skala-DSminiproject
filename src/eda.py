"""DAY1: read only required HDF5 fields, keeping raw measurements intact."""
from pathlib import Path
import json, re, hashlib, warnings
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from scipy.stats import spearmanr

FILES = {'Batch 1':'2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         'Batch 2':'2018-02-20_batchdata_updated_struct_errorcorrect.mat',
         'Batch 3':'2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
FEATURES = ['log_var_delta_q','min_delta_q','mean_delta_q','qd_slope_10_100',
            'mean_qd','std_qd','mean_ir','delta_ir','mean_tavg','mean_tmax',
            'mean_chargetime','c1','soc_switch','c2','early_positive_current']
COLORS = {'Batch 1':'#2364aa','Batch 2':'#d96839','Batch 3':'#36997b'}

def vector(f, obj):
    if isinstance(obj, h5py.Reference): obj = f[obj]
    a = obj[()]
    if h5py.check_dtype(ref=a.dtype):
        return np.concatenate([np.asarray(f[r][()]).reshape(-1) for r in a.ravel() if r])
    return np.asarray(a).reshape(-1).astype(float)

def at(f, group, name, i):
    refs = group[name]
    r = refs[i,0] if refs.shape[0] > 1 else refs[0,i]
    return f[r]

def get_curve(f, g, name, i):
    if name not in g: return np.array([])
    d = g[name]
    refs = d[()].reshape(-1)
    return vector(f, refs[i]) if i < len(refs) and refs[i] else np.array([])

def clean_summary(a, name):
    a = a.astype(float).copy()
    a[~np.isfinite(a)] = np.nan
    if name in ['QD','QC']: a[(a <= 0) | (a > 1.32)] = np.nan
    if name == 'IR': a[a <= 0] = np.nan
    if name in ['Tavg','Tmax','Tmin']: a[(a <= 0) | (a > 100)] = np.nan
    if name == 'chargetime': a[a <= 0] = np.nan
    return a

def knee_candidate(x,y):
    """Exploratory continuous two-line fit; full-life values never model inputs."""
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 10) & (y >= .80) & (y <= 1.32)
    x,y = x[keep],y[keep]
    if len(x) < 100: return (np.nan,np.nan,np.nan)
    y = median_filter(y,size=11,mode='nearest')
    take = np.unique(np.linspace(0,len(x)-1,min(300,len(x))).astype(int))
    x,y = x[take],y[take]
    t = (x-x[0])/(x[-1]-x[0]); ones=np.ones(len(t))
    base = np.column_stack([ones,t]); b=np.linalg.lstsq(base,y,rcond=None)[0]
    sse0=np.sum((y-base@b)**2)
    best=None
    for k in np.linspace(.15,.85,71):
        A=np.column_stack([ones,t,np.maximum(t-k,0)])
        coef=np.linalg.lstsq(A,y,rcond=None)[0]; sse=np.sum((y-A@coef)**2)
        if best is None or sse < best[0]: best=(sse,k,coef)
    sse,k,coef=best
    improvement=1-sse/max(sse0,1e-12)
    s1,s2=coef[1],coef[1]+coef[2]
    ratio=abs(s2)/max(abs(s1),1e-6)
    accepted=improvement>=.20 and s2 < 0 and abs(s2)>1.5*abs(s1)
    return (x[0]+k*(x[-1]-x[0]) if accepted else np.nan,improvement,ratio)

def extract(raw_dir, out_dir):
    raw_dir,out_dir=Path(raw_dir),Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    records=[]; summaries=[]; delta_curves={}; q10_curves={}; q100_curves={}; voltage_curves={}; audits=[]
    mapping={'QD':'QDischarge','QC':'QCharge','IR':'IR','Tavg':'Tavg',
             'Tmax':'Tmax','Tmin':'Tmin','chargetime':'chargetime'}
    for label,filename in FILES.items():
        path=raw_dir/filename
        if not path.exists(): raise FileNotFoundError(f'{label} missing: {path}')
        with h5py.File(path,'r') as f:
            b=f['batch']; n=b['summary'].size
            print(f'{label}: {n} cells, reading {path.name}',flush=True)
            for i in range(n):
                cid=f'b{label[-1]}c{i}'; sg=at(f,b,'summary',i); cg=at(f,b,'cycles',i)
                volt=vector(f,at(f,b,'Vdlin',i))
                life=float(vector(f,at(f,b,'cycle_life',i))[0])
                pol=at(f,b,'policy_readable',i)[()].reshape(-1)
                policy=''.join(chr(int(v)) for v in pol if v)
                fields={k:vector(f,sg[v]) for k,v in mapping.items() if v in sg}
                cyc=vector(f,sg['cycle']); length=len(cyc)
                if any(len(v)!=length for v in fields.values()):
                    raise ValueError(f'{cid}: summary field lengths disagree')
                cfields={k:clean_summary(v,k) for k,v in fields.items()}
                frame=pd.DataFrame({'cell_id':cid,'batch':label,'cycle':cyc,
                                    'cycle_life':life,'charging_policy':policy,**fields})
                summaries.append(frame)
                r={'cell_id':cid,'batch':label,'raw_cell_index':i,'cycle_life':life,
                   'charging_policy':policy,'n_summary_cycles':length,
                   'first_cycle':cyc[0],'last_cycle':cyc[-1]}
                early=(cyc>=1)&(cyc<=100)
                for k,a in cfields.items():
                    z=a[early]; z=z[np.isfinite(z)]
                    r['mean_'+k.lower()]=float(z.mean()) if len(z) else np.nan
                    r['std_'+k.lower()]=float(z.std(ddof=1)) if len(z)>1 else np.nan
                    r['invalid_'+k]=int(np.sum(~np.isfinite(a)))
                q=cfields['QD']; valid=np.isfinite(q)
                r['qd_last_valid']=float(q[valid][-1]) if valid.any() else np.nan
                eol=np.flatnonzero(valid&(q<.88)&(cyc>=10))
                r['observed_eol_cycle']=float(cyc[eol[0]]) if len(eol) else np.nan
                # These exports often stop just BEFORE the first QD<0.88 cycle.
                # Following the original MATLAB loader's 0.885 end-capacity check,
                # mark terminal QD>0.885 (or missing) as uncertain completion.
                r['eol_not_observed']=not np.isfinite(r['qd_last_valid']) or r['qd_last_valid']>.885
                r['eol_below_088_observed']=bool(len(eol))
                r['label_missing']=not np.isfinite(life)
                fit=(cyc>=10)&(cyc<=100)&valid
                r['qd_slope_10_100']=float(np.polyfit(cyc[fit],q[fit],1)[0]) if fit.sum()>=20 else np.nan
                ir=cfields['IR']; lo=ir[(cyc>=1)&(cyc<=10)]; hi=ir[(cyc>=91)&(cyc<=100)]
                r['delta_ir']=float(np.nanmedian(hi)-np.nanmedian(lo)) if np.isfinite(lo).any() and np.isfinite(hi).any() else np.nan
                kn,impr,ratio=knee_candidate(cyc,q)
                r.update(knee_cycle=kn,knee_fit_improvement=impr,knee_slope_ratio=ratio)
                parts=re.search(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C',policy)
                r.update(c1=float(parts[1]) if parts else np.nan,
                         soc_switch=float(parts[2])/100 if parts else np.nan,
                         c2=float(parts[3]) if parts else np.nan)
                # Match the recorded cycle number to its internal-cycle reference.
                # Confirmed against schema/summary lengths; cycle 1 is not assumed to be array 0.
                indices={int(v):j for j,v in enumerate(cyc) if np.isfinite(v)}
                count=cg['Qdlin'].size
                r['cycle_reference_count']=count
                if count!=length: raise ValueError(f'{cid}: cycle/summary alignment requires review')
                cur=[]
                for c in range(1,6):
                    if c in indices:
                        a=get_curve(f,cg,'I',indices[c]); pos=a[np.isfinite(a)&(a>0.1)]
                        if len(pos): cur.append(np.median(pos))
                r['early_positive_current']=float(np.mean(cur)) if cur else np.nan
                qs=[]
                for c in [10,100]:
                    qs.append(get_curve(f,cg,'Qdlin',indices[c]) if c in indices else np.array([]))
                a,z=qs
                r['q10_points']=len(a);r['q100_points']=len(z)
                r['delta_available']=len(a)==len(z)==len(volt)==1000
                r['log_var_delta_q']=r['min_delta_q']=r['mean_delta_q']=np.nan
                if r['delta_available']:
                    d=z-a; good=np.isfinite(a)&np.isfinite(z)&np.isfinite(volt)&(a>=-.05)&(z>=-.05)&(a<=1.32)&(z<=1.32)
                    d[~good]=np.nan; r['delta_valid_points']=int(good.sum())
                    if good.sum()>=900 and np.nanmax(a)-np.nanmin(a)>.5 and np.nanmax(z)-np.nanmin(z)>.5:
                        r['log_var_delta_q']=float(np.log10(max(np.nanvar(d,ddof=1),1e-16)))
                        r['min_delta_q']=float(np.nanmin(d)); r['mean_delta_q']=float(np.nanmean(d))
                        delta_curves[cid]=d; q10_curves[cid]=a; q100_curves[cid]=z
                        voltage_curves[cid]=volt
                    else:r['delta_available']=False
                # Descriptive labels, not the binary modeling target.
                r['eda_life_group']='Unknown' if not np.isfinite(life) else ('Long >1000' if life>1000 else ('Short <500' if life<500 else 'Middle'))
                r['label550']=int(life>=550) if np.isfinite(life) else np.nan
                records.append(r)
            audits.append({'batch':label,'file':filename,'bytes':path.stat().st_size,'raw_cells':n})
    cells=pd.DataFrame(records); summary=pd.concat(summaries,ignore_index=True)
    cells.to_csv(out_dir/'cell_features.csv',index=False)
    summary.to_csv(out_dir/'cycle_summary.csv',index=False)
    np.savez_compressed(out_dir/'delta_curves.npz',**delta_curves)
    np.savez_compressed(out_dir/'q10_curves.npz',**q10_curves)
    np.savez_compressed(out_dir/'q100_curves.npz',**q100_curves)
    np.savez_compressed(out_dir/'voltage_curves.npz',**voltage_curves)
    (out_dir/'data_manifest.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    return cells,summary,delta_curves

def finish(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

def plots(cells,summary,curves,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=list(FILES)
    with np.load(out.parent/'data/processed/voltage_curves.npz') as z:
        voltage_curves={k:z[k] for k in z.files}
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,label in zip(axes,groups):
        sub=cells[cells.batch==label]
        ax.hist(sub.cycle_life,bins=np.r_[np.arange(150,2300,200),2300],color=COLORS[label],edgecolor='white')
        ax.axvline(550,color='#555',ls='--',label='Binary threshold 550'); ax.axvline(sub.cycle_life.median(),color='#8b254f',ls=':',label='Median')
        ax.set(title=f'{label} | labeled={sub.cycle_life.notna().sum()}/{len(sub)}',xlabel='Total cycle life',ylabel='Cells',xlim=(150,2300));ax.legend(fontsize=8)
    finish(fig,out/'01_life_distribution.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained'); norm=Normalize(cells.cycle_life.min(),cells.cycle_life.max())
    for ax,label in zip(axes,groups):
        for _,r in cells[cells.batch==label].iterrows():
            sub=summary[summary.cell_id==r.cell_id]; q=clean_summary(sub.QD.to_numpy(),'QD')
            ax.plot(sub.cycle,q,color=plt.cm.viridis(norm(r.cycle_life)) if np.isfinite(r.cycle_life) else '#b8b8b8',alpha=.6,lw=.65)
        ax.axhline(.88,color='#b43f45',ls='--',lw=1);ax.set(title=label,xlabel='Recorded cycle',ylabel='Discharge capacity (Ah)',ylim=(.7,1.23))
    sm=plt.cm.ScalarMappable(norm=norm,cmap='viridis'); fig.colorbar(sm,ax=axes,label='Total cycle life',shrink=.8)
    finish(fig,out/'02_degradation.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,label in zip(axes,groups):
        for _,r in cells[cells.batch==label].iterrows():
            if r.cell_id in curves:
                ax.plot(voltage_curves[r.cell_id],curves[r.cell_id],color=plt.cm.viridis(norm(r.cycle_life)) if np.isfinite(r.cycle_life) else '#b8b8b8',alpha=.6,lw=.65)
        ax.set(title=label,xlabel='Recorded Vdlin (V)',ylabel='Q100(V) - Q10(V) (Ah)')
    finish(fig,out/'03_delta_q.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for label in groups:
        sub=cells[cells.batch==label]
        axes[0].scatter(sub.log_var_delta_q,sub.cycle_life,label=label,c=COLORS[label],alpha=.8,s=27)
        axes[1].scatter(sub.c1,sub.cycle_life,label=label,c=COLORS[label],alpha=.65,s=27)
    axes[0].set(xlabel='log10 variance of Delta Q',ylabel='Total cycle life',title='Early degradation signature')
    axes[1].set(xlabel='Stage-1 C-rate',ylabel='Total cycle life',title='Charging policy and life (association)')
    axes[0].legend();axes[1].legend();finish(fig,out/'04_early_signals.png')
    agg=cells.groupby(['batch','charging_policy']).cycle_life.agg(['mean','std','count']).reset_index()
    agg.to_csv(out/'policy_statistics.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    for ax,label in zip(axes,groups):
        sub=agg[(agg.batch==label)&(agg['count']>0)].sort_values('mean',ascending=False);x=np.arange(len(sub))
        ax.bar(x,sub['mean'],yerr=sub['std'].fillna(0),color=COLORS[label],capsize=2)
        ax.set_xticks(x,sub.charging_policy,rotation=65,ha='right',fontsize=7)
        for xx,(_,r) in zip(x,sub.iterrows()):ax.text(xx,r['mean']+(r['std'] if np.isfinite(r['std']) else 0)+25,f'n={r["count"]}',ha='center',fontsize=6)
        ax.set(title=label+' | mean +/- SD (n=1: SD unavailable)',ylabel='Total cycle life')
    finish(fig,out/'05_policy.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for label in groups:
        sub=cells[cells.batch==label]
        axes[0].scatter(sub.early_positive_current,sub.qd_slope_10_100,label=label,c=COLORS[label],alpha=.7,s=27)
        axes[1].scatter(sub.mean_chargetime,sub.cycle_life,label=label,c=COLORS[label],alpha=.7,s=27)
    axes[0].set(xlabel='Initial positive-current median (A)',ylabel='QD slope, cycles 10-100 (Ah/cycle)',title='Measured current and early degradation')
    axes[1].set(xlabel='Mean charge time, cycles 1-100 (min)',ylabel='Total cycle life',title='Charge time and life')
    axes[0].legend();axes[1].legend();finish(fig,out/'10_charging_measurements.png')
    heat=['log_var_delta_q','qd_slope_10_100','mean_ir','mean_tavg','mean_tmax','mean_chargetime','cycle_life']
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,label in zip(axes,groups):
        corr=cells[cells.batch==label][heat].corr(method='spearman')
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(heat)),[x.replace('mean_','') for x in heat],rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(len(heat)),[x.replace('mean_','') for x in heat],fontsize=8);ax.set_title(label)
        for i in range(len(heat)):
            for j in range(len(heat)):
                v=corr.iloc[i,j]
                if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>.65 else '#222')
    fig.colorbar(im,ax=axes,shrink=.75,label='Spearman rho');finish(fig,out/'06_correlations.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,label in zip(axes,groups):
        sub=cells[(cells.batch==label)&cells.knee_cycle.notna()]
        ax.scatter(sub.cycle_life,sub.knee_cycle,color=COLORS[label]);ax.plot([0,2500],[0,2500],ls=':',color='#777')
        ax.set(title=f'{label} | candidates {len(sub)}',xlabel='Total cycle life',ylabel='Exploratory knee cycle')
    finish(fig,out/'07_knee_candidates.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,label in zip(axes,groups):
        s=cells[cells.batch==label]
        ids=[s.loc[s.cycle_life.idxmin(),'cell_id'],s.iloc[(s.cycle_life-s.cycle_life.median()).abs().argmin()].cell_id,s.loc[s.cycle_life.idxmax(),'cell_id']]
        for cid in ids:
            if cid in curves:
                ax.plot(voltage_curves[cid],curves[cid],label=f'{cid}: {cells.set_index("cell_id").loc[cid,"cycle_life"]:.0f} cycles')
        ax.set(title=label,xlabel='Recorded Vdlin (V)',ylabel='Delta Q (Ah)');ax.legend(fontsize=7)
    finish(fig,out/'08_delta_examples.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,label in zip(axes,groups):
        s=cells[(cells.batch==label)&cells.knee_cycle.notna()].sort_values('cycle_life')
        if len(s):
            r=s.iloc[len(s)//2];ss=summary[summary.cell_id==r.cell_id]
            ax.plot(ss.cycle,clean_summary(ss.QD.to_numpy(),'QD'),c=COLORS[label],lw=1)
            ax.axvline(r.knee_cycle,c='#b43f45',ls='--',label=f'Candidate ~{r.knee_cycle:.0f}')
            ax.set_title(f'{label}: {r.cell_id}');ax.legend(fontsize=8)
        ax.axhline(.88,c='#777',ls=':');ax.set(xlabel='Recorded cycle',ylabel='QD (Ah)',ylim=(.7,1.2))
    finish(fig,out/'09_knee_examples.png')

def tables(cells,out):
    out=Path(out)
    stats=[];corrs=[]
    for label,s in cells.groupby('batch',sort=False):
        stats.append({'batch':label,'n':len(s),'n_labeled':int(s.cycle_life.notna().sum()),'label_missing':int(s.cycle_life.isna().sum()),'min':s.cycle_life.min(),'median':s.cycle_life.median(),
                      'mean':s.cycle_life.mean(),'max':s.cycle_life.max(),
                      'short_lt500':int((s.cycle_life<500).sum()),'long_gt1000':int((s.cycle_life>1000).sum()),
                      'label0_lt550':int((s.cycle_life<550).sum()),'label1_ge550':int((s.cycle_life>=550).sum()),
                      'eol_not_observed':int(s.eol_not_observed.sum()),'delta_usable':int(s.delta_available.sum()),
                      'knee_candidates':int(s.knee_cycle.notna().sum()),'policy_count':s.charging_policy.nunique()})
        for feature in FEATURES:
            pair=s[[feature,'cycle_life']].dropna()
            if len(pair)>=4 and pair[feature].nunique()>1:
                rho,p=spearmanr(pair[feature],pair.cycle_life)
                corrs.append({'batch':label,'feature':feature,'n':len(pair),'spearman':rho,'p_unadjusted':p,
                              'pearson':pair[feature].corr(pair.cycle_life)})
    pd.DataFrame(stats).to_csv(out/'batch_statistics.csv',index=False)
    pd.DataFrame(corrs).to_csv(out/'feature_correlations.csv',index=False)
    cells.nsmallest(10,'cycle_life').to_csv(out/'shortest_cells.csv',index=False)
    cells[cells.eol_not_observed|~cells.delta_available|cells.cycle_life.isna()].to_csv(out/'quality_review_cells.csv',index=False)
    eligible=cells[(cells.batch=='Batch 1')&~cells.eol_not_observed&cells.cycle_life.notna()]
    eligible[FEATURES+['cycle_life']].corr(method='spearman')['cycle_life'].drop('cycle_life').rename('spearman_terminal_screened').to_csv(out/'batch1_eol_sensitivity.csv')

def run(root):
    root=Path(root)
    cells,summary,curves=extract(root/'data/raw',root/'data/processed')
    plots(cells,summary,curves,root/'results');tables(cells,root/'results')
    print(pd.read_csv(root/'results/batch_statistics.csv').to_string(index=False))
    return cells,summary,curves

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=p.parse_args()
    run(args.root)
