"""DAY 1 robustness, time-weighted early current features and fixed split plan.
No estimator is trained; full-life knee data never enter model inputs.
"""
from pathlib import Path
import argparse,json
import h5py,numpy as np,pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eda
CURRENT=['charge_mean_a','charge_rms_a','charge_p95_a','charge_fraction_gt4a']
INPUTS=['log_var_delta_q','qd_slope_10_100','mean_chargetime','c1','mean_tavg']

def interval_features(t,current):
    """Time-weighted interval approximation; exclude zero/negative dt and discharge."""
    t=np.asarray(t);current=np.asarray(current)
    if len(t)!=len(current) or len(t)<2:return None
    dt=np.diff(t);level=(current[:-1]+current[1:])/2
    good=np.isfinite(dt)&np.isfinite(level)&(dt>0)&(current[:-1]>.1)&(current[1:]>.1)
    if not good.any():return None
    w=dt[good];v=level[good];order=np.argsort(v)
    p95=v[order][np.searchsorted(np.cumsum(w[order]),.95*w.sum())]
    return dict(zip(CURRENT,[np.average(v,weights=w),np.sqrt(np.average(v*v,weights=w)),p95,w[v>4].sum()/w.sum()]))

def extract_current(root,raw_dir):
    rows=[]
    for batch,filename in eda.FILES.items():
        path=Path(raw_dir)/filename
        with h5py.File(path,'r') as h:
            b=h['batch']
            for i in range(b['summary'].size):
                sg=eda.at(h,b,'summary',i);cg=eda.at(h,b,'cycles',i)
                cycles=eda.vector(h,sg['cycle']);assert len(cycles)==cg['I'].size==cg['t'].size
                features=[];used=[]
                for cycle in range(2,6):
                    ix=np.flatnonzero(cycles==cycle)
                    if not len(ix):continue
                    j=int(ix[0]);x=interval_features(eda.get_curve(h,cg,'t',j),eda.get_curve(h,cg,'I',j))
                    if x is not None:features.append(x);used.append(cycle)
                rows.append({'cell_id':f'b{batch[-1]}c{i}','batch':batch,'cycle_window':'2-5','early_cycles_used':len(used),'max_cycle_used':max(used) if used else np.nan,**{k:np.mean([f[k] for f in features]) if features else np.nan for k in CURRENT}})
        print(batch,'early current extracted',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(Path(root)/'data/processed/early_current_features.csv',index=False)
    return frame

def knee_fit(x,y,smooth):
    good=np.isfinite(x)&np.isfinite(y)&(x>=10)&(y>=.8)&(y<=1.32);x=x[good];y=y[good]
    if len(x)<100:return np.nan,np.nan,np.nan,False
    y=median_filter(y,size=smooth,mode='nearest');idx=np.unique(np.linspace(0,len(x)-1,min(300,len(x))).astype(int));x=x[idx];y=y[idx]
    t=(x-x[0])/(x[-1]-x[0]);base=np.column_stack([np.ones(len(t)),t]);coef=np.linalg.lstsq(base,y,rcond=None)[0];sse0=np.sum((y-base@coef)**2)
    best=(np.inf,None,None)
    for k in np.linspace(.15,.85,71):
        a=np.column_stack([base,np.maximum(t-k,0)]);co=np.linalg.lstsq(a,y,rcond=None)[0];sse=np.sum((y-a@co)**2)
        if sse<best[0]:best=sse,k,co
    sse,k,co=best;s1=co[1];s2=co[1]+co[2]
    return x[0]+k*(x[-1]-x[0]),1-sse/max(sse0,1e-12),abs(s2)/max(abs(s1),1e-6),s2<0

def knee_sensitivity(root,cells,summary):
    rows=[]
    for cid,g in summary.groupby('cell_id',sort=False):
        x=g.cycle.to_numpy();y=g.QD.to_numpy()
        for smooth in [7,11,21]:
            pos,gain,ratio,negative=knee_fit(x,y,smooth)
            for threshold in [.1,.2,.3]:
                for slope_ratio in [1.2,1.5,2.]:
                    accepted=negative and gain>=threshold and ratio>slope_ratio
                    rows.append({'cell_id':cid,'smooth':smooth,'min_gain':threshold,'min_ratio':slope_ratio,'fitted_knee_cycle':pos,'accepted':accepted,'knee_cycle':pos if accepted else np.nan})
    grid=pd.DataFrame(rows);grid.to_csv(Path(root)/'results/knee_sensitivity_grid.csv',index=False)
    stats=[]
    for cid,g in grid.groupby('cell_id',sort=False):
        ref=cells.set_index('cell_id').loc[cid];valid=g.knee_cycle.dropna()
        stats.append({'cell_id':cid,'batch':ref.batch,'accepted_settings':len(valid),'settings':len(g),'acceptance_fraction':len(valid)/len(g),'position_span_fraction_record':(valid.max()-valid.min())/(ref.last_cycle-10) if len(valid) else np.nan})
    stability=pd.DataFrame(stats);stability.to_csv(Path(root)/'results/knee_stability_cells.csv',index=False)
    grouped=stability.groupby('batch').agg(n=('cell_id','size'),all_settings_detected=('accepted_settings',lambda s:(s==27).sum()),median_acceptance=('acceptance_fraction','median'),median_position_span=('position_span_fraction_record','median'),max_position_span=('position_span_fraction_record','max')).reset_index()
    grouped.to_csv(Path(root)/'results/knee_stability_summary.csv',index=False)
    return grouped

def cluster_bootstrap_spearman(frame,feature,n_resamples=2000,seed=42):
    """Resample complete charging-policy groups, retaining within-group cells.

    Percentile intervals describe this observed batch under group exchangeability;
    they are exploratory, not multiplicity-adjusted or model-performance intervals.
    """
    keys=['c1','soc_switch','c2']
    cols=list(dict.fromkeys([feature,'cycle_life']+keys))
    data=frame[cols].replace([np.inf,-np.inf],np.nan).dropna()
    groups=[g[[feature,'cycle_life']].to_numpy() for _,g in data.groupby(keys,sort=True)]
    if len(groups)<2:raise ValueError('At least two complete policy groups are required')
    if data[feature].nunique()<2 or data.cycle_life.nunique()<2:
        raise ValueError('Spearman correlation requires variation in both variables')
    point=float(spearmanr(data[feature],data.cycle_life).statistic)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(n_resamples):
        sample=np.concatenate([groups[j] for j in rng.integers(0,len(groups),size=len(groups))])
        if np.ptp(sample[:,0])==0 or np.ptp(sample[:,1])==0:continue
        rho=float(spearmanr(sample[:,0],sample[:,1]).statistic)
        if np.isfinite(rho):draws.append(rho)
    if len(draws)<.9*n_resamples:raise ValueError('Too many degenerate bootstrap samples')
    lo,hi=np.quantile(draws,[.025,.975])
    return {'feature':feature,'n':len(data),'policy_groups':len(groups),'spearman':point,
            'ci_low':float(lo),'ci_high':float(hi),'bootstrap_requested':n_resamples,
            'bootstrap_valid':len(draws),'seed':seed,'method':'policy_cluster_percentile_95'}

def correlation_uncertainty(root,merged):
    rows=[]
    for batch,g in merged.groupby('batch',sort=True):
        for feature in ['log_var_delta_q','charge_rms_a']:
            rows.append({'batch':batch,**cluster_bootstrap_spearman(g,feature)})
    ci=pd.DataFrame(rows);ci.to_csv(Path(root)/'results/correlation_uncertainty.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,3.8));fig.subplots_adjust(left=.35,right=.77,top=.90,bottom=.17)
    labels=[]
    for i,row in ci.iterrows():
        color=eda.COLORS[row.batch]
        ax.hlines(i,row.ci_low,row.ci_high,color=color,lw=2)
        ax.scatter(row.spearman,i,color=color,s=28,zorder=3)
        name='log var Delta Q' if row.feature=='log_var_delta_q' else 'RMS current'
        labels.append(f'{row.batch} | {name} (n={row.n}, K={row.policy_groups})')
        ax.text(1.025,i,f'{row.spearman:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]',
                transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.set_yticks(range(len(ci)),labels,fontsize=9)
    ax.invert_yaxis();ax.axvline(0,color='#777',lw=.8,ls='--')
    ax.set(xlim=(-1.05,1.05),xlabel='Spearman correlation with cycle life',title='Exploratory 95% intervals: 2,000 policy-cluster resamples')
    ax.grid(axis='x',alpha=.2);fig.savefig(Path(root)/'results/16_correlation_uncertainty.png',dpi=180);plt.close(fig)
    return ci

def model_inputs(frame,features):
    """Explicit early-only whitelist; diagnostic/target columns cannot enter X."""
    allowed=set(eda.FEATURES+CURRENT)
    if not features or len(set(features))!=len(features) or not set(features)<=allowed:
        raise ValueError('Only unique early-cycle features from the whitelist are allowed')
    return frame.loc[:,features].copy()

def split_plan(root,cells):
    eligible=cells[np.isfinite(cells.cycle_life) & (cells.cycle_life>0) & ~cells.eol_not_observed].copy()
    b1=eligible[eligible.batch=='Batch 1'].sort_values('cell_id').reset_index(drop=True)
    assert model_inputs(b1,INPUTS).notna().all().all()
    groups=b1[['c1','soc_switch','c2']].apply(lambda r:'|'.join(f'{v:g}' for v in r),axis=1)
    train,valid=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(b1,groups=groups))
    assert not set(groups.iloc[train])&set(groups.iloc[valid])
    assignment=b1[['cell_id','batch','cycle_life']].copy();assignment['policy_group']=groups;assignment['partition']='train_cv';assignment.loc[valid,'partition']='holdout';assignment['cv_fold']=pd.Series(pd.NA,index=assignment.index,dtype='Int64')
    folds=[]
    for fold,(a,b) in enumerate(GroupKFold(5).split(b1.iloc[train],groups=groups.iloc[train]),1):
        a=train[a];b=train[b];assert not set(groups.iloc[a])&set(groups.iloc[b]);assignment.loc[b,'cv_fold']=fold
        folds.append({'fold':fold,'train_cells':len(a),'valid_cells':len(b),'train_groups':groups.iloc[a].nunique(),'valid_groups':groups.iloc[b].nunique(),'policy_overlap':0})
    cohorts=[('batch1_eda',cells[cells.batch=='Batch 1']),('batch1_screened',b1),
             ('train_cv',b1.iloc[train]),('holdout',b1.iloc[valid])]
    ranges=pd.DataFrame([{'cohort':name,'n':len(g),'life_min':g.cycle_life.min(),
                         'life_max':g.cycle_life.max(),'life_median':g.cycle_life.median()}
                        for name,g in cohorts])
    ranges.to_csv(Path(root)/'results/target_range_by_cohort.csv',index=False)
    lo=b1.iloc[train].cycle_life.min();hi=b1.iloc[train].cycle_life.max()
    coverage=[]
    for batch in ['Batch 2','Batch 3']:
        g=eligible[eligible.batch==batch]
        coverage.append({'batch':batch,'n':len(g),'train_life_min':lo,'train_life_max':hi,
                         'below_train_target':int((g.cycle_life<lo).sum()),
                         'above_train_target':int((g.cycle_life>hi).sum())})
    pd.DataFrame(coverage).to_csv(Path(root)/'results/external_target_coverage.csv',index=False)
    assignment.to_csv(Path(root)/'data/processed/batch1_split_plan.csv',index=False)
    pd.DataFrame(folds).to_csv(Path(root)/'results/cv_split_plan.csv',index=False)
    info={'label_policy':'Finite cycle_life and terminal QD <=0.885 Ah; no terminal QD is input. Same screen across batches.','batch1_main_n':len(b1),'batch1_sensitivity_n':46,'batch1_groups':groups.nunique(),'train_n':len(train),'holdout_n':len(valid),'train_groups':groups.iloc[train].nunique(),'holdout_groups':groups.iloc[valid].nunique(),'cv_folds':5,'seed':42,'policy_overlap':0,'train_target_min':float(lo),'train_target_max':float(hi),'test2_n':len(eligible[eligible.batch=='Batch 2']),'optional_test3_n':len(eligible[eligible.batch=='Batch 3']),'input_whitelist':eda.FEATURES+CURRENT,'primary_feature_sequence':INPUTS,'current_extension':['charge_rms_a'],'models_trained':False}
    (Path(root)/'data/processed/modeling_plan.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    return info

def run(root,raw_dir=None):
    root=Path(root);cells=pd.read_csv(root/'data/processed/cell_features.csv');summary=pd.read_csv(root/'data/processed/cycle_summary.csv')
    path=root/'data/processed/early_current_features.csv'
    cur=pd.read_csv(path) if path.exists() and raw_dir is None else extract_current(root,raw_dir or root/'data/raw')
    merged=cells.merge(cur.drop(columns='batch'),on='cell_id',validate='one_to_one');assert len(merged)==139 and (cur.max_cycle_used<=5).all()
    cor=[]
    for batch,g in merged.groupby('batch'):
        for k in CURRENT:
            for target in ['cycle_life','qd_slope_10_100','log_var_delta_q','c1']:
                v=g[[k,target]].dropna();cor.append({'batch':batch,'feature':k,'target':target,'n':len(v),'spearman':v[k].corr(v[target],method='spearman')})
    correlations=pd.DataFrame(cor);correlations.to_csv(root/'results/current_pattern_correlations.csv',index=False)
    stability=knee_sensitivity(root,cells,summary);plan=split_plan(root,cells)
    intervals=correlation_uncertainty(root,merged)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,(batch,g) in zip(axes,merged.groupby('batch')):
        v=g.dropna(subset=['cycle_life']);ax.scatter(v.charge_rms_a,v.cycle_life,color=eda.COLORS[batch],s=20);ax.set(title=batch,xlabel='Time-weighted RMS current (A), cycles 2-5',ylabel='Cycle life')
    fig.savefig(root/'results/14_current_pattern_population.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');st=pd.read_csv(root/'results/knee_stability_cells.csv')
    axes[0].bar(stability.batch,stability.all_settings_detected);axes[0].set(ylabel='Detected in all 27 settings (cells)',title='Acceptance across 3 fits x 9 threshold rules')
    axes[1].boxplot([st[st.batch==b].position_span_fraction_record.dropna() for b in eda.FILES],tick_labels=list(eda.FILES));axes[1].set(ylabel='Knee position span / record length',title='Location spread among accepted fits (1% grid)')
    fig.savefig(root/'results/15_knee_sensitivity.png',dpi=180);plt.close(fig)
    print(stability.to_string(index=False));print(json.dumps(plan,ensure_ascii=False,indent=2));print(correlations.to_string(index=False))
    return {'stability':stability,'plan':plan,'correlations':correlations,'current':cur,'correlation_intervals':intervals}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--raw-dir');args=p.parse_args();run(args.root,args.raw_dir)
