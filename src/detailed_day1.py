"""DAY1 evidence-driven report: measured findings -> decisions, with no model fit."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,PageBreak

FONT='/System/Library/Fonts/Supplemental/AppleGothic.ttf'
GROUPS=['Batch 1','Batch 2','Batch 3']
CORE=['log_var_delta_q']
EXTENDED=['qd_slope_10_100','mean_chargetime','c1','mean_tavg']

def ensure_current_profiles(root):
 import h5py
 root=Path(root);path=root/'data/processed/current_profile_examples.npz'
 if path.exists():return
 cells=pd.read_csv(root/'data/processed/cell_features.csv');manifest=json.loads((root/'data/processed/data_manifest.json').read_text());arrays={};rows=[]
 def vec(h,d):
  a=d[()]
  if h5py.check_dtype(ref=a.dtype):a=np.concatenate([h[r][()].ravel() for r in a.ravel()])
  return np.asarray(a,dtype=float).ravel()
 for m in manifest:
  group=cells[(cells.batch==m['batch'])&cells.cycle_life.notna()&~cells.eol_not_observed]
  with h5py.File(root/'data/raw'/m['file'],'r') as h:
   b=h['batch']
   for label,r in [('Lower-life example',group.loc[group.cycle_life.idxmin()]),('Higher-life example',group.loc[group.cycle_life.idxmax()])]:
    i=int(r.raw_cell_index);sg=h[b['summary'][i,0]];cg=h[b['cycles'][i,0]];cyc=vec(h,sg['cycle']);j=int(np.flatnonzero(cyc==5)[0])
    I=vec(h,h[cg['I'][()].ravel()[j]]);t=vec(h,h[cg['t'][()].ravel()[j]])
    assert len(t)==len(I) and np.isfinite(t).all() and t[-1]>t[0]
    arrays[r.cell_id+'_x']=(t-t[0])/(t[-1]-t[0]);arrays[r.cell_id+'_I']=I
    rows.append({'cell_id':r.cell_id,'batch':r.batch,'example_group':label,'cycle_life':r.cycle_life,'charging_policy':r.charging_policy,'cycle':5,'points':len(I),'positive_current_max':np.max(I)})
 np.savez_compressed(path,**arrays);pd.DataFrame(rows).to_csv(root/'data/processed/current_profile_examples.csv',index=False)

def analyze(root):
 root=Path(root);f=pd.read_csv(root/'data/processed/cell_features.csv');s=pd.read_csv(root/'data/processed/cycle_summary.csv')
 corr=pd.read_csv(root/'results/feature_correlations.csv');stats=pd.read_csv(root/'results/batch_statistics.csv');out=root/'results'
 quality=[];groups=[];knees=[];outliers=[];redundancy=[]
 for b,g in f.groupby('batch',sort=False):
  y=g.cycle_life.dropna();q1,q3=y.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
  flagged=g[(g.cycle_life<lo)|(g.cycle_life>hi)].copy();flagged['tukey_low']=lo;flagged['tukey_high']=hi;outliers.append(flagged)
  quality.append({'batch':b,'raw_n':len(g),'target_n':len(y),'target_missing':g.cycle_life.isna().sum(),'completion_uncertain':g.eol_not_observed.sum(),'eol_crossing_equals_label':np.isclose(g.cycle_life,g.observed_eol_cycle).sum(),'label_equals_last_plus_one':np.isclose(g.cycle_life,g.last_cycle+1).sum(),'target_skew':y.skew(),'log_target_skew':np.log(y).skew(),'tukey_low':lo,'tukey_high':hi,'tukey_short_count':(g.cycle_life<lo).sum(),'tukey_long_count':(g.cycle_life>hi).sum(),'positive_early_slope':(g.qd_slope_10_100>0).sum()})
  for name in ['Short <500','Middle','Long >1000']:
   x=g[g.eda_life_group==name]
   groups.append({'batch':b,'life_group':name,'n':len(x),**{k:x[k].median() for k in ['log_var_delta_q','min_delta_q','mean_delta_q']}})
  rel=g.knee_cycle/g.cycle_life
  knees.append({'batch':b,'candidates':g.knee_cycle.notna().sum(),'knee_cycle_median':g.knee_cycle.median(),'relative_knee_median_labeled':rel.median(),'relative_n':rel.notna().sum(),'current_vs_slope_rho':g.early_positive_current.corr(g.qd_slope_10_100,method='spearman')})
  for a,z in [('log_var_delta_q','min_delta_q'),('log_var_delta_q','mean_delta_q'),('min_delta_q','mean_delta_q'),('mean_tavg','mean_tmax'),('c1','soc_switch')]:
   pair=g[[a,z]].dropna();redundancy.append({'batch':b,'feature1':a,'feature2':z,'n':len(pair),'spearman':pair[a].corr(pair[z],method='spearman')})
 quality=pd.DataFrame(quality);groupstats=pd.DataFrame(groups);knee=pd.DataFrame(knees);redundancy=pd.DataFrame(redundancy)
 quality.to_csv(out/'detailed_quality_statistics.csv',index=False);groupstats.to_csv(out/'delta_life_group_statistics.csv',index=False);knee.to_csv(out/'knee_batch_comparison.csv',index=False);redundancy.to_csv(out/'feature_redundancy.csv',index=False)
 pd.concat(outliers).to_csv(out/'tukey_review_cells.csv',index=False)
 # Labelled life groups, not misleading minimum-vs-maximum proxies.
 with np.load(root/'data/processed/delta_curves.npz') as dz,np.load(root/'data/processed/voltage_curves.npz') as vz:
  fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
  for ax,b in zip(axes,GROUPS):
   for name,color in [('Short <500','#c2504b'),('Middle','#6c7f93'),('Long >1000','#148b72')]:
    ids=f[(f.batch==b)&(f.eda_life_group==name)&f.delta_available].cell_id.tolist()
    if not ids:continue
    vs=np.stack([vz[c] for c in ids]);assert np.allclose(vs,vs[0])
    arr=np.stack([dz[c] for c in ids]);med=np.nanmedian(arr,axis=0);a,z=np.nanpercentile(arr,[25,75],axis=0)
    ax.plot(vs[0],med,color=color,label=f'{name}: n={len(ids)}');ax.fill_between(vs[0],a,z,color=color,alpha=.16)
   if not ((f.batch==b)&(f.eda_life_group=='Short <500')).any():ax.text(.02,.03,'No short-life (<500) cells',transform=ax.transAxes,fontsize=8)
   ax.set(title=b,xlabel='Recorded voltage (V)',ylabel='Delta Q (Ah)');ax.legend(fontsize=8)
  fig.savefig(out/'11_delta_life_groups.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
 for i,b in enumerate(GROUPS):
  y=f.loc[f.batch==b,'cycle_life'].dropna();axes[0,i].hist(y,bins=10,color='#287e93',edgecolor='white');axes[1,i].hist(np.log(y),bins=10,color='#9a6378',edgecolor='white')
  axes[0,i].set(title=f'{b}: skew={y.skew():.2f}',xlabel='Cycle life',ylabel='Cells');axes[1,i].set(title=f'log life: skew={np.log(y).skew():.2f}',xlabel='Natural log cycle life',ylabel='Cells')
 fig.savefig(out/'12_target_transformation.png',dpi=180,bbox_inches='tight');plt.close(fig)
 ensure_current_profiles(root)
 profiles=pd.read_csv(root/'data/processed/current_profile_examples.csv')
 with np.load(root/'data/processed/current_profile_examples.npz') as z:
  fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
  for ax,b in zip(axes,GROUPS):
   for (_,r),color in zip(profiles[profiles.batch==b].iterrows(),['#c2504b','#148b72']):
    ax.plot(z[r.cell_id+'_x'],z[r.cell_id+'_I'],color=color,lw=1.2,label=f'{r.cell_id}: life {r.cycle_life:.0f}')
   ax.axhline(0,color='#888',ls=':',lw=.8);ax.set(title=b,xlabel='Normalized elapsed time, cycle 5',ylabel='Recorded current I (A)');ax.legend(fontsize=8)
  fig.savefig(out/'13_current_profiles.png',dpi=180,bbox_inches='tight');plt.close(fig)
 return {'root':root,'cells':f,'summary':s,'corr':corr,'stats':stats,'quality':quality,'groupstats':groupstats,'knee':knee,'redundancy':redundancy}

def manuscript(c):
 f=c['cells'];stats=c['stats'].set_index('batch');quality=c['quality'].set_index('batch');corr=c['corr'].pivot(index='feature',columns='batch',values='spearman');gs=c['groupstats'];kn=c['knee'].set_index('batch');red=c['redundancy']
 pages=[];current=None
 def page(key,title):
  nonlocal current
  current={'key':key,'title':title,'blocks':[]};pages.append(current)
 def para(label,text):current['blocks'].append(('p',label,text))
 def tbl(headers,rows,widths=None):current['blocks'].append(('t',headers,rows,widths))
 def img(name,height=180):current['blocks'].append(('i',name,height))
 def rho(k,b):return f'{corr.loc[k,b]:+.3f}'
 page('overview','초기 배터리 신호로 총수명을 예측하는 모델 전략')
 para('DAY 1 분석 목적','처음 100사이클에서 확보할 수 있는 신호로 배터리 셀의 총수명을 예측하는 회귀 전략을 수립한다. 세 배치의 EDA를 통해 핵심 신호, 배치 차이, 라벨 신뢰성과 모델 복잡도의 한계를 확인했다. 모델 학습·성능 평가는 이번 보고서의 범위가 아니다.')
 tbl(['정의','본 프로젝트의 선택'],[['분석 단위','배터리 셀 1개 = 모델 표본 1개. 116,722개 사이클 행을 독립 표본으로 취급하지 않는다.'],['X: 예측 시점의 입력','초기 100사이클까지의 방전곡선 변화, 용량 추세, 충전 조건, 온도·저항'],['Y: Target','원본 cycle_life: SOH 80%에 해당하는 총수명(사이클). 100사이클 시점 RUL과 구분.'],['핵심 선택','ΔQ log 분산으로 작은 기본 모델을 시작하고, 용량 기울기·충전시간·C-rate·온도를 단계적으로 추가한다.'],['검증 원칙','Batch 1 내부 CV·Hold-out으로 선택. Batch 2 필수 외부 평가, Batch 3 추가 평가.']], [100,390])
 para('한눈에 보는 핵심 발견','전체 139개 셀 중 수명 라벨은 129개다. ΔQ log 분산과 수명의 상관은 Batch 1/2/3에서 -0.871/-0.709/-0.797로 방향이 일관된다. 충전시간과 초기 용량 기울기는 배치별 방향이 달라 조건 의존성이 크다. 따라서 모든 후보를 한 번에 넣는 전략을 피한다.')
 para('참고 이미지의 적용 방식','이미지의 매출·날씨·Tweedie 예시를 그대로 가져오지 않고, 탐색 목적 → 실제 관찰 → 처리 이유 → 처리 전후 확인 → 피처 및 후보 모델 선택의 논리 구조를 배터리 데이터에 적용했다. 캠퍼스·반·팀원 이름은 제출 전에 입력한다.')
 page('data','입력 데이터·수집 구조·라벨 품질')
 tbl(['배치','전체 / 라벨','결측 라벨','완료 불확실','파일 날짜'],[[b,f'{int(stats.loc[b,"n"])} / {int(stats.loc[b,"n_labeled"])}',int(stats.loc[b,'label_missing']),int(stats.loc[b,'eol_not_observed']),date] for b,date in zip(GROUPS,['2017-05-12','2018-02-20','2018-04-12'])],[75,100,90,95,130])
 para('수집·정리','Kaggle의 실습 지정 MAT 파일을 HDF5로 읽었다. 셀 스칼라(cycle_life·policy), summary(사이클별 용량·저항·온도·충전시간), cycles(사이클 내부 곡선)를 연결했다. 배치+원본 인덱스로 cell_id를 만들고 실제 사이클 번호와 곡선 참조를 대응시켰다. 원본 summary는 보존했다.')
 para('처리 전후','피처 계산에서 용량 ≤0 또는 >1.32 Ah, IR ≤0, 온도 ≤0 또는 >100°C, 충전시간 ≤0을 결측 처리했다. 셀 139개는 탐색용으로 유지하고 라벨 통계에서는 결측 10개만 제외했다. 완료 불확실 16개와 결측 10개는 겹치는 목록이므로 합산하지 않는다.')
 para('배치별 종료 방식의 차이','Batch 1의 46개와 Batch 3의 유효 라벨 44개는 cycle_life=마지막 기록 사이클+1이다. Batch 2의 유효 라벨 39개는 최초 QD<0.88 Ah 사이클과 cycle_life가 일치하고, 이후 측정도 남아 있다. 따라서 마지막 기록 사이클을 모든 배치의 타깃으로 쓰면 잘못된다.')
 para('완료 불확실 기준과 누락','종료 시 유효 QD>0.885 Ah 또는 미제공을 완료 불확실로 표시했다. 단일 기준은 EOL의 확정 판정이 아니다. Batch 1의 일부 종료 용량은 0.91~1.04 Ah여서 라벨이 실제 EOL보다 관측 종료를 나타낼 가능성을 검토해야 한다. 원본 3개 모두 초기 10·100사이클의 유효 ΔQ 곡선은 확보됐다.')
 para('데이터 출처의 차이','실습 Batch 2는 2018-02-20이며 연구진 공개 로더의 2017-06-30과 다르다. 공개 코드의 병합·제외 인덱스를 그대로 적용하지 않았다. 상세 원본 대조와 결측/품질 목록은 results의 검증 파일로 제공한다.')
 page('q1','Q1. Cycle Life 분포: 배치 차이는 무엇인가?')
 para('탐색 목적','학습 범위와 외부 평가 범위가 얼마나 다른지 확인하고, 짧은 수명이 일부 오류인지 일반적인 배치 특성인지 구분한다. 150~2,300사이클의 같은 축으로 비교한다.')
 img('01_life_distribution.png',165)
 tbl(['배치','중앙값 / 평균','범위','<500','>1000'],[[b,f'{stats.loc[b,"median"]:.1f} / {stats.loc[b,"mean"]:.1f}',f'{stats.loc[b,"min"]:.0f}~{stats.loc[b,"max"]:.0f}',f'{int(stats.loc[b,"short_lt500"])} ({stats.loc[b,"short_lt500"]/stats.loc[b,"n_labeled"]:.1%})',f'{int(stats.loc[b,"long_gt1000"])} ({stats.loc[b,"long_gt1000"]/stats.loc[b,"n_labeled"]:.1%})'] for b in GROUPS],[70,125,85,105,105])
 para('관찰·배치 비교','Batch 1은 중앙값 858.5로 중간 수명에 집중한다. Batch 2는 중앙값 472.0이고 71.8%가 500 미만으로, 단수명이 지배적이다. Batch 3는 중앙값 1005.5이고 52.3%가 1000 초과다. 세 배치 모두 오른쪽 꼬리가 있으나 Batch 2·3에서 더 두드러진다.')
 para('시사점 → 전략','Batch 1의 유효 수명 범위는 534~1227이다. Batch 2의 <534 구간과 Batch 3의 >1227 구간은 학습 타깃 범위 밖이므로 해당 구간의 잔차와 MAPE를 별도로 점검한다. 이것이 입력 X의 외삽을 확정하는 것은 아니며, 실제 입력 피처의 지원 범위도 함께 확인해야 한다.')
 para('분모와 그룹 정의','비율의 분모는 수명 라벨이 있는 셀 46/39/44개다. EDA의 장·단수명 기준(>1000/<500)과 분류 과제의 기준(≥550/<550)은 다른 정의다. 500~1000은 중간 그룹이다.')
 page('q1_outlier','Q1. 짧은 셀은 왜 짧은가? 이상치와 구분')
 tbl(['배치','배치 내 최단 셀','수명','충전 정책','종료 QD'],[[b,r.cell_id,f'{r.cycle_life:.0f}',r.charging_policy,f'{r.qd_last_valid:.3f} Ah'] for b in GROUPS for _,r in f[f.batch==b].nsmallest(1,'cycle_life').iterrows()],[65,90,60,190,85])
 para('통계적 이상치 확인','배치별 1.5×IQR 기준에서 낮은 수명 이상치는 모두 0개다. 짧은 셀을 자동으로 이상치라고 부르면 안 된다. 높은 수명 후보는 Batch 1 0개, Batch 2 9개, Batch 3 3개다. 분포의 꼬리라는 이유만으로 정상 장수명 셀을 삭제하지 않는다.')
 r=f[f.cell_id=='b2c19'].iloc[0]
 para('짧은 셀의 근거: b2c19',f'수명 392사이클, 정책 6C(60%)-3C다. 최초 QD<0.88 Ah 사이클도 392로 타깃과 일치한다. ΔQ log 분산은 {r.log_var_delta_q:.3f}로 Batch 2 장수명 그룹 중앙값 -4.271보다 크다. 실제 초기 곡선 변화가 큰 짧은 셀이며, 종료 기록 오류만으로 짧아진 사례라는 근거는 없다.')
 para('프로토콜 대조','같은 6C(60%)-3C 정책의 평균 수명은 400.0±11.3(n=2)이다. 다른 짧은 정책인 3.6C(9%)-5C도 394.5±2.1(n=2)로 짧다. 첫 단계 C-rate가 낮아도 뒤 단계가 높을 수 있어 “첫 단계가 빠르기 때문에 짧다”는 단일 설명은 부족하다.')
 para('완료 불확실 셀과의 차이','예를 들어 b1c0는 라벨 1190이지만 마지막 용량 1.026 Ah로 아직 0.88 Ah에 도달하지 않았다. 이런 라벨 신뢰성 문제와 실제 단수명을 분리해야 한다. 높은 수명 정책 평균에는 완료 불확실 셀이 포함될 수 있으므로 그대로 정책 우열을 결론내리지 않는다.')
 para('시사점 → 처리','짧은 정상 셀을 삭제하지 않고, 결측 라벨·종료 불확실·측정 오류를 각각 플래그로 관리한다. 짧은 수명의 확정 원인은 실험 통제나 추가 메타데이터 없이는 규명할 수 없다. ΔQ·후반 충전 단계·온도 차이는 검증할 가설이다.')
 page('q2','Q2. 방전용량은 일정하게 감소하는가?')
 img('02_degradation.png',195)
 para('관찰·배치 비교','세 배치에서 완만한 초기 감소 뒤 더 급격해지는 말기 구간이 보인다. Batch 2는 상대적으로 짧은 기록과 빠른 종료가 많고, Batch 3는 긴 사이클 동안 완만하게 유지되는 셀이 많다. 색은 실제 수명이며 회색은 결측 라벨이다. 초기 용량 수준만으로 전체 수명을 구분하기 어렵다.')
 tbl(['배치','초기 평균 QD의 셀 평균','초기 기울기 >0인 셀','초기 QD 기울기 vs 수명 ρ'],[[b,f'{f[f.batch==b].mean_qd.mean():.3f} Ah',f'{int(quality.loc[b,"positive_early_slope"])} / {int(stats.loc[b,"n"])}',rho('qd_slope_10_100',b)] for b in GROUPS],[65,150,125,150])
 para('초기 기울기의 해석','초기 10~100사이클 용량이 증가하는 셀도 Batch 1/2/3에서 12/23/1개다. 초기 안정화나 측정 특성의 가능성이 있어 기울기가 항상 열화 속도를 뜻한다고 단정할 수 없다. 특히 수명 상관이 +0.578/-0.271/+0.184로 바뀌므로 ΔQ와 같은 범용 신호로 취급하지 않는다.')
 para('시사점 → 피처','초기 100사이클의 용량 기울기·평균·변동성은 후보로 유지하지만, ΔQ 단일 모델에 추가했을 때 Batch 1 검증 성능이 개선되는지 확인한다. 전체 궤적이나 말기 감소율을 입력으로 쓰면 100사이클 예측 시점을 어긴다.')
 para('그래프 해석의 한계','말기 QD 0.88 Ah 선은 공칭 1.1 Ah의 80%다. Batch 2는 라벨 이후 측정도 포함하고 Batch 1·3는 EOL 직전 종료가 흔하므로, 곡선 길이만으로 배치의 실제 열화 특성을 판단하지 않는다.')
 page('q2_knee','Q2. Knee point: 가속 열화의 탐색적 후보')
 img('09_knee_examples.png',175)
 tbl(['배치','후보 / 전체','knee 사이클 중앙값','knee / 수명 중앙값²'],[[b,f'{int(kn.loc[b,"candidates"])} / {int(stats.loc[b,"n"])}',f'{kn.loc[b,"knee_cycle_median"]:.1f}',f'{kn.loc[b,"relative_knee_median_labeled"]:.1%} (n={int(kn.loc[b,"relative_n"])})'] for b in GROUPS],[70,105,150,165])
 para('방법','10사이클 이후 QD 0.80~1.32 Ah의 유효점에 11점 중앙값 평활화를 적용한다. 기록 구간의 15~85%에서 연속 두 직선의 전환점을 탐색한다. 단일 직선 대비 SSE 20% 이상 개선, 후반 기울기 음수, 후반 절댓값이 전반의 1.5배 초과일 때 후보로 표시한다.')
 para('발견과 한계','후보가 45/44/46개로 대부분의 셀에서 탐지된다. 이는 곡률에 민감한 탐색 규칙이며 모든 셀에 물리적 knee가 확정됐다는 뜻이 아니다. 위치는 평활화·탐색 범위·측정 종료에 의존한다. Batch 2의 라벨 이후 기록도 이 탐색에 포함한다.')
 para('시사점 → 모델 설계','대체로 기록 후반의 가속 열화가 초기 정보와 다르다는 점을 확인했다. knee 위치는 수명 전체가 있어야 계산되므로 입력에서 제외한다. 100사이클 이전의 국소 추세만 후보로 사용하고, 물리적 knee 검증은 별도 과제로 둔다.')
 para('통계 분모','² 셀별 knee/cycle_life를 계산한 뒤 중앙값을 낸 값이다. 결측 라벨은 비율에서 제외하며, 후보 사이클 중앙값은 라벨 없는 셀도 포함한 후보 전체 기준이다. 따라서 두 요약의 분모가 다르다.')
 page('q3','Q3. ΔQ(V): 장·단수명 그룹의 실제 차이')
 para('계산 정의','실제 사이클 번호 100과 10을 선택하고, 동일 Vdlin 축에서 ΔQ(V)=Q100(V)-Q10(V)를 계산했다. 시간축이 다른 원시 Qd를 직접 빼지 않는다. 전압은 가이드의 설명 범위 대신 실제 Vdlin 3.5→2.0 V를 사용했다.')
 img('11_delta_life_groups.png',195)
 para('그래프 읽기','선은 그룹별 중앙값, 음영은 25~75% 범위다. 그룹 내부 변동도 함께 보이므로 최단·최장 셀 한 쌍만 비교할 때보다 대표성이 높다. 다만 Batch 2 장수명은 n=3으로 요약 곡선의 안정성은 제한적이다. Batch 2의 단수명 그룹은 약 3 V 부근에서 더 깊은 음의 변화가 보이며 장수명 그룹은 변화가 작다.')
 para('존재하지 않는 그룹을 만들지 않기','Batch 1과 Batch 3에는 <500사이클 단수명 셀이 없다. 따라서 이 두 배치는 중간(500~1000)과 장수명(>1000)을 비교했다. 기존 최단/중앙/최장 예시 셀 그래프를 보조자료로 유지하되 최단 셀을 <500 그룹이라고 부르지 않는다.')
 para('시사점 → 피처','곡선의 변화 크기와 퍼짐을 log 분산·최솟값·평균으로 요약한다. 전체 1000개 전압 포인트를 한 번에 입력하면 46개 학습 셀보다 차원이 훨씬 커져 과적합 가능성이 높다. 작은 요약 피처부터 검증한다.')
 page('q3_stats','Q3. 곡선에서 추출한 통계값과 피처 선택')
 tbl(['배치 / 그룹','n','log10 분산 중앙값','ΔQ 최솟값 중앙값','ΔQ 평균 중앙값'],[[r.batch+' / '+{'Short <500':'단수명','Middle':'중간','Long >1000':'장수명'}[r.life_group],int(r.n),f'{r.log_var_delta_q:.3f}',f'{r.min_delta_q:.4f}',f'{r.mean_delta_q:.4f}'] for _,r in gs[gs.n>0].iterrows()],[115,35,125,115,100])
 para('발견','Batch 2의 단수명/장수명 log 분산 중앙값은 -3.446/-4.271이다. 최솟값은 -0.0562/-0.0192 Ah로 단수명에서 더 깊다. Batch 1과 Batch 3에서도 중간 대비 장수명 그룹의 log 분산과 용량 변화 절댓값이 작다.')
 para('피처 정의','log_var_delta_q=log10(var(ΔQ, ddof=1)), min_delta_q=min(ΔQ), mean_delta_q=mean(ΔQ)다. 로그는 작은 분산의 크기 차이를 비교하기 쉽도록 적용했으며 로그 자체가 예측 성능을 보장하지 않는다. 1000포인트 중 유효점 900개 이상 등 곡선 품질 조건을 확인했다.')
 para('선정과 중복 제어','세 배치에서 log 분산과 수명 관계가 강하고 방향이 일치하므로 핵심 피처로 우선 선택한다. 최솟값·평균은 대체 요약량 또는 정규화 모델의 후보로 비교한다. ΔQ 세 변수를 무조건 동시에 넣지 않는다.')
 para('추가 검증','Batch 1에서 완료 불확실 10개를 제외해도 log 분산 상관은 -0.871(n=46)→-0.826(n=36)으로 유지된다. 전압 구간별 피처를 추가할 경우 선택은 Batch 1 학습 fold 안에서만 수행하며 외부 타깃으로 최적화하지 않는다.')
 page('q4_policy','Q4. 충전 프로토콜별 수명: 빠를수록 짧은가?')
 img('05_policy.png',330)
 para('관찰','Batch 1에서 5.4C(80%)-5.4C는 평균 546.5(n=2), 8C(35%)-3.6C는 607.5(n=2)다. 더 빠른 첫 단계 정책이 항상 더 짧지 않다. Batch 3의 4.8C(80%)-4.8C는 평균 1564.2(n=6)로 긴 편이지만 다른 프로토콜·전환 SOC·배치 조건도 함께 다르다.')
 para('그룹 크기와 라벨 영향','막대는 평균, 에러바는 표준편차이며 신뢰구간이 아니다. n=1은 표준편차 추정 불가다. Batch 1의 상위 평균 정책에는 완료 불확실 셀이 있어 우열을 확정할 수 없다. Batch 2에서 유효 라벨이 없는 정책은 그래프에서 제외하고 정책표에는 남겼다.')
 para('시사점 → 피처·모델','정책을 c1·soc_switch·c2로 분해하고 실제 충전시간·전류를 함께 검토한다. 첫 단계 C-rate 하나로 수명을 설명하지 않는다. 정책별 타깃 평균을 그대로 입력하는 target encoding은 작은 그룹과 누수 위험이 커서 초기 모델에서 사용하지 않는다.')
 page('q4_profiles','Q4. 실제 전류 파형: 단계별 조건을 구분')
 img('13_current_profiles.png',195)
 profiles=pd.read_csv(c['root']/'data/processed/current_profile_examples.csv')
 tbl(['배치 / 셀','수명','충전 프로토콜','5사이클 최대 I'],[[r.batch+' / '+r.cell_id,f'{r.cycle_life:.0f}',r.charging_policy,f'{r.positive_current_max:.2f} A'] for _,r in profiles.iterrows()],[110,55,225,100])
 para('예시 선정과 축','완료 불확실 플래그가 없는 유효 라벨 셀 중 배치별 최소·최대 수명 셀을 선택했다. 이는 그룹 전체의 대표성을 보장하는 표본 추출이 아니라 실제 파형을 설명하는 6개 예시다. 실제 cycle 5의 I(t)를 읽고 시간은 각 사이클의 경과시간을 0~1로 정규화했다. 양수는 충전, 음수는 방전이다.')
 para('발견','Batch 3의 b3c28은 3.7C(31%)-5.9C 정책으로, 첫 단계가 낮아도 뒤 단계에서 기록 최대 I가 5.90 A다. 더 긴 b3c38은 5C(67%)-4C, 최대 5.00 A다. 단계별 세기와 전환점이 중요하며 c1 하나로 전체 충전 조건을 표현하기 어렵다.')
 para('시사점 → 패턴 피처','전체 양의 전류 중앙값은 오래 지속되는 낮은 전류 구간에 영향을 받아 빠른 단계의 세기를 놓칠 수 있다. 현재 요약값의 상관은 다음 페이지에서 비교한다. 추가 패턴 피처로 초기 충전 전류 상위 분위수·단계별 체류시간을 검토할 수 있으나, 아직 전체 셀에서 추출·검증한 최종 입력으로 주장하지 않는다.')
 page('q4_current','Q4. 실측 전류 패턴과 열화 속도의 관계')
 img('10_charging_measurements.png',195)
 tbl(['배치','실측 전류 vs 초기 QD 기울기 ρ','c1 vs 수명 ρ','충전시간 vs 수명 ρ'],[[b,f'{kn.loc[b,"current_vs_slope_rho"]:+.3f}',rho('c1',b),rho('mean_chargetime',b)] for b in GROUPS],[70,175,115,130])
 para('어떤 전류를 비교했는가?','초기 1~5사이클에서 실측 I>0.1 A의 중앙값을 구하고 셀별 평균했다. 전체 전류 패턴의 요약값이며 단계별 체류시간·펄스·전환 구간을 모두 표현하지는 않는다. 정책 C-rate와 측정된 A 단위 전류는 구분했다. 전류 대 기울기의 n은 전체 46/47/46개, 수명 상관의 n은 유효 라벨 46/39/44개다.')
 para('관찰·배치 비교','전류와 초기 기울기의 상관은 +0.408/-0.444/-0.103으로 방향이 다르다. c1과 수명도 -0.483/+0.055/-0.229다. 충전시간은 +0.613/-0.387/+0.196으로 바뀐다. 이 자료만으로 고속 충전의 단일한 수명 효과를 확정할 수 없다.')
 para('시사점 → 검증','충전 조건 피처는 조건 의존적인 추가 후보로 다룬다. Batch 1에서 핵심 ΔQ 모델에 하나씩 추가해 개선 여부를 확인하고, 최종 평가에서 배치 차이를 해석한다. 개선되지 않으면 제외한다. c2·전환 SOC와의 상호작용은 제한된 복잡도로만 검토한다.')
 page('q5','Q5. 어떤 초기 신호가 수명과 연결되는가?')
 img('04_early_signals.png',165)
 keys=['log_var_delta_q','min_delta_q','mean_delta_q','mean_chargetime','qd_slope_10_100','c1','mean_tavg','mean_ir']
 tbl(['초기 피처','Batch 1 ρ (n)','Batch 2 ρ (n)','Batch 3 ρ (n)'],[[k,*[rho(k,b)+f' ({int(c["corr"][(c["corr"].batch==b)&(c["corr"].feature==k)].n.iloc[0])})' for b in GROUPS]] for k in keys],[190,100,100,100])
 para('가장 강한 관계','각 배치에서 제시한 초기 후보 중 ΔQ log 분산의 절대 상관이 가장 크다. 최솟값·평균도 방향이 일관된다. 초기 평균 용량은 +0.226/+0.112/+0.162로 상대적으로 약해, 초기 수준 자체보다 변화량이 더 유용한 후보라는 근거가 된다.')
 para('배치 간 안정성','온도와 충전시간·기울기의 관계는 약해지거나 부호가 바뀐다. Batch 1의 상관 순위만으로 범용 피처라고 주장하지 않는다. 외부 배치의 관찰은 조건 의존성을 설명하는 데 사용하며 모델 선택·튜닝은 Batch 1 안에서 한다.')
 para('통계 해석','Spearman은 단조 관계, Pearson은 선형 관계를 확인한다. 상관은 탐색적 근거로, 예측 성능·인과·독립적인 효과를 입증하지 않는다. 결측을 쌍별로 제외하고 n을 기록했다. 여러 피처의 p-value는 다중 비교 보정 없는 탐색 결과다.')
 page('q5_redundancy','Q5. 다중공선성: 강한 신호를 중복 넣지 않기')
 img('06_correlations.png',190)
 pairs=[('log_var_delta_q','min_delta_q'),('log_var_delta_q','mean_delta_q'),('min_delta_q','mean_delta_q'),('mean_tavg','mean_tmax'),('c1','soc_switch')]
 tbl(['변수 쌍','Batch 1','Batch 2','Batch 3'],[[a+' / '+z,*[f'{red[(red.batch==b)&(red.feature1==a)&(red.feature2==z)].spearman.iloc[0]:+.3f}' for b in GROUPS]] for a,z in pairs],[250,80,80,80])
 para('관찰','ΔQ 요약량 세 개는 상호 |ρ|가 약 0.95~0.99로 거의 같은 정보를 담는다. Tavg/Tmax는 0.951/0.759/0.975로 중복되고 Batch 1의 c1/전환 SOC도 -0.869로 연결된다. 이 값은 단조 중복의 근거이며 선형 공선성을 정량화하는 VIF와 동일하지 않다.')
 para('모델 영향 → 대응','일반 선형 회귀에서 중복 피처는 계수를 불안정하게 만들 수 있다. 초기 모델은 ΔQ log 분산 하나와 평균 온도 하나를 사용하고, Ridge/ElasticNet으로 축소 또는 선택을 검토한다. 트리는 계수 문제는 다르지만 중복 피처의 중요도가 분산될 수 있어 해석을 주의한다.')
 para('검증 기준','피처 제거·선택과 스케일링은 CV 학습 fold에서만 수행한다. 필요하면 그 fold에서 Pearson 공선성·VIF도 확인한다. 중복량을 늘린 모델이 검증 평균뿐 아니라 fold 간 변동도 개선하는지 비교한다.')
 page('features','Feature Engineering: 가설에서 선택 기준까지')
 tbl(['피처 묶음','사용할 입력','선정 이유 / 확인할 사항'],[['G0 핵심 기본형','log_var_delta_q','세 배치 일관된 방향, 완료 불확실 제외에도 관계 유지. 먼저 이 신호만으로 기준을 만든다.'],['G1 초기 추세','G0 + qd_slope_10_100','열화 곡선의 국소 추세. 배치별 부호 차이 때문에 추가 효과를 검증한다.'],['G2 충전 조건','G1 + mean_chargetime + c1','정책·실측 조건 보완. 충전 조건이 바뀌는 환경에서도 일반화 가능한지는 미확정.'],['G3 열적 조건','G2 + mean_tavg','온도 영향 후보. Tmax는 중복이 커서 동시에 쓰지 않는다.'],['대체 / 확장 후보','min_delta_q, mean_delta_q, IR 변화, c2, 전환 SOC, 실측 전류','핵심 ΔQ 요약량 교체 또는 정규화 비교. 추가 변수가 필요하다는 근거는 CV로 확인.'],['입력 제외','cycle_life, label550, last_cycle, 전체 knee, 종료 QD, observed_eol_cycle, 배치 ID·셀 ID','타깃 또는 예측 시점 이후 정보, 식별자. 완료 플래그는 라벨 검토용이며 X가 아니다.']],[100,160,230])
 para('선별 절차','G0→G1→G2→G3를 동일한 Batch 1 분할로 비교한다. 후보 집합은 좁게 유지하고 추가로 얻는 CV MAPE 개선과 fold 간 안정성을 확인한다. 개선이 작거나 불안정하면 더 단순한 집합을 선택한다. 실제 최종 집합은 DAY 2 검증 후 확정한다.')
 para('예측 시점과 결측','summary 평균은 cycle≤100, 기울기는 10~100, IR 변화는 1~10 대비 91~100, 실측 전류는 1~5를 사용한다. 결측 대체는 학습 fold 중앙값으로 한다. 결측 표시 피처를 추가할 경우에도 초기 관측에서 계산한 표시만 허용한다.')
 para('참고 이미지와의 연결','입력 변수의 출처와 가설, 타깃과 입력의 구분, 시계열에서 미래값을 사용하지 않는 기준을 명시했다. 원논문 피처를 무조건 복제하지 않고 이번 데이터의 배치 차이와 라벨 품질을 반영했다.')
 page('task','Regression 선택과 Target 변환 검토')
 tbl(['항목','회귀: 이번 선택','분류: 선택하지 않은 이유'],[['예측 시점 / 타깃','100사이클 / cycle_life 수치','5사이클 / life≥550이면 1'],['정보 활용','ΔQ100−10과 100사이클 요약 사용 가능','ΔQ100−10을 사용하면 시점 위반'],['Batch 1 분포','연속 수명 534~1227 활용','클래스 0/1 = 1/45로 검증 매우 불안정'],['업무 해석','셀별 예상 사이클 수·선별 우선순위','550 기준 통과 여부로 정보가 압축됨']],[100,195,195])
 img('12_target_transformation.png',210)
 para('변환 전후 확인',f'원 수명→자연로그 수명의 왜도는 Batch 1 {quality.loc["Batch 1","target_skew"]:.2f}→{quality.loc["Batch 1","log_target_skew"]:.2f}, Batch 2 {quality.loc["Batch 2","target_skew"]:.2f}→{quality.loc["Batch 2","log_target_skew"]:.2f}, Batch 3 {quality.loc["Batch 3","target_skew"]:.2f}→{quality.loc["Batch 3","log_target_skew"]:.2f}다. 로그 후 일부 비대칭은 줄지만 모든 배치가 정규분포가 되는 것은 아니다.')
 para('시사점 → 타깃 처리','Batch 1에서 원 타깃과 log 타깃을 제한된 후보로 비교한다. log 모델은 예측 후 exp로 사이클 단위로 복원하고 원 단위 MAPE·MAE·RMSE로 평가한다. MAPE 최적과 log 제곱오차 최적은 같지 않으므로 변환을 자동 확정하지 않는다. 매출 예시의 Tweedie 분포를 배터리 수명에 그대로 적용하지 않는다.')
 page('models','Modeling Strategy: 데이터 특성에 맞는 후보 모델')
 tbl(['후보','EDA 근거와 알고리즘 관점','복잡도·선택 조건'],[['중앙값 기준 모델','작은 표본에서도 해석 가능한 비교 기준. 배치별 수명 범위 차이의 영향을 확인.','모든 학습 fold의 라벨 중앙값만 사용. 학습 데이터 밖 타깃을 참조하지 않는다.'],['Ridge / ElasticNet','ΔQ 요약량·온도 중복과 46개 이하의 작은 학습 표본. 계수를 정규화해 분산을 줄인다.','중앙값 대체→표준화→정규화 회귀. alpha는 작은 로그 간격 후보, ElasticNet 혼합비도 좁게 비교.'],['Random Forest','충전 조건·온도와 곡선 변화의 비선형 상호작용 가능성. 분할 평균으로 비선형을 표현한다.','깊이·최소 리프 표본을 제한. 타깃 학습 범위 밖으로 직접 외삽하기 어려워 장수명 구간 오차를 확인.'],['얕은 Gradient Boosting','단순 모델에 남는 비선형 잔차를 작은 트리로 순차 보완한다.','깊이 1~2, 적은 트리·낮은 학습률을 제한적으로 비교. 작은 표본에서 튜닝 폭을 넓히지 않는다.']],[105,220,165])
 para('왜 딥러닝·1000차원 곡선 모델을 우선 쓰지 않는가?','학습 독립 표본은 Batch 1 최대 46개이며 라벨 검토 후 더 줄 수 있다. 116,722개 사이클 행이 독립 셀 표본 수를 늘려주지는 않는다. 많은 파라미터나 곡선 포인트로 시작하면 표본 대비 복잡도가 커지고 검증이 불안정해진다.')
 para('LightGBM을 반드시 써야 하는가?','참고 이미지의 LightGBM은 매출 데이터의 구체적 선택 예시다. 본 프로젝트에서는 부스팅 계열을 후보로 두되 현재 표본 수에서 정규화 회귀보다 유리하다는 근거가 없다. 추가 라이브러리나 최신 알고리즘 자체를 선정 이유로 삼지 않는다.')
 para('최종 선택 원칙','CV 평균 MAPE가 낮고 fold 간 변동이 작은 후보를 우선한다. 성능 차이가 미미하면 피처 수와 복잡도가 작은 모델을 선택한다. 정규화 회귀도 외삽이 항상 정확한 것은 아니므로 Batch 2·3 평가에서 실제 한계를 보고한다.')
 page('validation','데이터 처리·검증·오류 분석의 구체적 계획')
 para('학습 라벨 품질의 기본안','Batch 1 완료 불확실 10개를 제외한 36개를 주 분석 대상으로 검토하고, 46개 전체를 사용한 민감도 비교를 별도로 보고한다. 제외는 명시한 종료 용량 기준에 따르며 결과를 보고 유리한 셀만 삭제하지 않는다. 어느 라벨 정책을 주 결과로 삼을지는 분할 전에 고정한다.')
 para('분할','셀 단위 Hold-out 약 20%를 random_state=42로 먼저 고정한다. 충전 프로토콜 그룹 분리를 우선 검토하고 그룹 수·학습 표본 수가 허용하면 학습 부분에서 GroupKFold 약 5분할을 사용한다. 그룹 분리가 불가능하면 셀 단위 분할의 한계를 공개한다. Hold-out은 피처·튜닝 선택에 사용하지 않는다.')
 para('Pipeline','각 CV 학습 fold에서 결측 중앙값 대체→선형 모델용 StandardScaler→모델 fit을 수행한다. 트리 모델은 불필요한 스케일링을 생략한다. 초기 측정치 기반 결측 지표는 선택 후보이며 전체 기록의 invalid 카운트는 입력에서 제외한다. 선택한 피처·파라미터·타깃 변환은 Hold-out 및 외부 평가 전에 고정한다.')
 para('외부 평가와 지표','Batch 2의 유효 라벨 39개로 최종 평가하되 명시한 품질 정책을 일관되게 적용하고 평가 n을 보고한다. Batch 3도 같은 정책으로 추가 평가한다. 주 지표 MAPE=100×mean(|y−ŷ|/y), 보조 MAE·RMSE를 사이클 단위로 보고한다. 타깃이 모두 양수여서 0 나눗셈은 없지만 작은 수명에 상대 오차가 더 민감하다.')
 para('오차를 어떻게 해석하는가?','CV 평균과 표준편차, Hold-out, Batch 2·3를 분리한다. Gap은 Valid−CV, Test−Valid, Test−9.1%(%p)로 정의한다. 큰 오차 셀의 수명 구간·충전 정책·초기 곡선·결측 패턴을 확인하고 편향을 설명한다. 이때 외부 타깃으로 다시 튜닝하지 않는다.')
 para('외부 EDA의 한계','DAY 1 요구에 따라 세 배치의 라벨을 이미 관찰했다. 외부 데이터를 완전히 눈가림했다고 주장하지 않는다. 배치별 관찰은 해석과 검증 위험 설명에 사용하고, 실질적인 모델·피처 선택은 Batch 1 학습 부분 안에서 수행한다.')
 page('mapping','EDA Insight → Feature → Modeling Strategy')
 tbl(['관찰한 사실','피처·처리 결정','후보 모델·검증으로 연결'],[['ΔQ log 분산\nρ=-0.871 / -0.709 / -0.797','핵심 G0 피처로 선정. 완료 불확실 제외 민감도 확인.','단일 피처 Ridge부터 시작하고 기준 모델 대비 개선 확인.'],['ΔQ 요약량 |ρ|≈0.95~0.99, 온도 중복 큼','요약량 하나·온도 하나 우선. 확장 시 정규화.','Ridge/ElasticNet 계수·fold 변동 확인.'],['충전시간·초기 기울기의 수명 상관 방향 변함','G1~G3로 하나씩 추가. 조건 의존 후보로 취급.','Batch 1 ablation 비교와 외부 배치 잔차 해석.'],['학습 셀 최대 46개, 라벨 품질 검토 시 36개','1000차원 곡선 대신 작은 요약 피처.','얕은 트리·좁은 튜닝. 딥러닝 우선 제외.'],['Batch 2 단수명 71.8%, Batch 3 장수명 52.3%','배치 ID를 모델 입력으로 넣지 않음. 타깃 구간별 평가.','학습 범위 밖 타깃과 입력 지원 범위를 분리해 점검.'],['초기 용량 일부 증가, knee는 전체 궤적에 의존','전체 knee·종료값 제외. 초기 국소 추세만 후보.','100사이클 시점 준수·누수 차단.'],['완료 불확실 16개, 라벨 결측 10개','별도 품질 플래그·사전 포함 기준.','라벨 검토·평가 n·민감도 분석 공개.']],[170,160,160])
 para('업무 관점의 활용','예상 총수명은 셀 선별이나 교체 우선순위의 참고 근거가 된다. 셀 간 비교와 초기 열화 신호 탐색이 목적이며, 현재 분석으로 실제 ESS 교체 날짜나 안전 제어를 직접 결정할 수는 없다. 운전 조건 변화·캘린더 열화·시간 단위 수명 변환·불확실성을 검증해야 한다.')
 page('coverage','요구사항 충족 확인·한계·참고 자료')
 tbl(['요구사항','보고서의 근거'],[['Q1 분포·비율·이상치','150~2300 히스토그램, 배치별 분모·비율, IQR와 짧은 실제 셀의 라벨·정책·ΔQ 대조'],['Q2 QD·가속 열화·knee','전체 곡선, 초기 기울기 양수 셀 수, knee 규칙·배치 비교·미래 정보 제외'],['Q3 ΔQ·그룹 비교·통계','실제 사이클 정렬, 그룹 중앙값·IQR 곡선, 없는 단수명 그룹 명시, 요약 피처 수치'],['Q4 정책·고속 충전·전류','정책 평균·SD·n, c1/실측 전류/충전시간의 배치별 상관, 인과 한계'],['Q5 상관·최강 신호·중복','세 배치 상관표, ΔQ 신호의 일관성과 민감도, 변수 중복과 정규화'],['모델 전략 3항목','X/Y와 회귀 선택, 피처 단계적 비교, EDA 근거별 후보 모델·Pipeline·분할 계획']],[160,330])
 para('남은 작업과 주장 범위','DAY 2에서 라벨 정책·분할을 고정하고 학습·CV·Hold-out·외부 평가를 수행한다. 지금의 상관과 왜도는 EDA 결과이며 모델 성능이 아니다. 원논문 9.1% MAPE를 달성했다고 보고하지 않는다. knee와 짧은 수명의 인과 원인은 확정하지 않았다.')
 para('재현 자료','notebooks/01_EDA.ipynb, src/eda.py, src/detailed_day1.py, results의 배치·그룹·품질·중복 통계, requirements.txt를 제공한다. 보고서 원고는 report/DAY1_모델전략_상세.md에 저장하여 수정·검토할 수 있도록 했다.')
 para('출처','실습 가이드: https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb\n데이터: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle\n연구진 로더:\nhttps://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation\n논문: Severson et al. (2019), Nature Energy, https://www.nature.com/articles/s41560-019-0356-8')
 return pages

def markdown(pages,images=True):
 lines=[]
 for page in pages:
  lines+=['## '+page['title'],'']
  for block in page['blocks']:
   if block[0]=='p':lines+=['### '+block[1],'',block[2],'']
   elif block[0]=='i' and images:lines+=['!['+block[1]+'](../results/'+block[1]+')','']
   elif block[0]=='t':
    lines+=['| '+' | '.join(map(str,block[1]))+' |','| '+' | '.join(['---']*len(block[1]))+' |']
    for row in block[2]:lines+=['| '+' | '.join(str(x).replace('|','/').replace('\n','<br>') for x in row)+' |']
    lines+=['']
 return '\n'.join(lines)

def base_compact_manuscript(c):
 """Twelve-page submission; detailed notebook and manuscript remain available."""
 old={p['key']:p for p in manuscript(c)};pages=[]
 def select(key,labels=None,images=True,tables=True,height=None):
  blocks=[]
  for b in old[key]['blocks']:
   if b[0]=='p' and (labels is None or b[1] in labels):blocks.append(b)
   elif b[0]=='t' and tables:blocks.append(b)
   elif b[0]=='i' and images:blocks.append((b[0],b[1],height or b[2]))
  return blocks
 def page(key,title,blocks):pages.append({'key':key,'title':title,'blocks':blocks})
 def para(label,text):return ('p',label,text)
 page('overview_data','분석 목적과 데이터 품질',
  select('overview',['DAY 1 분석 목적','한눈에 보는 핵심 발견'])+
  select('data',[],images=False)+[
  para('수집·처리','Kaggle 지정 MAT의 셀·summary·cycles를 실제 사이클 번호로 연결했다. 피처 계산에서 용량 ≤0 또는 >1.32 Ah, IR ≤0, 온도 ≤0 또는 >100°C, 충전시간 ≤0은 결측 처리하고 원본 summary는 보존했다. 결측 라벨 10개와 완료 불확실 16개는 겹치므로 합산하지 않는다.'),
  para('라벨 신뢰성과 출처','Batch 1·3의 유효 라벨은 마지막 기록+1, Batch 2는 최초 QD<0.88 Ah 사이클이다. 종료 QD>0.885 Ah 또는 미제공은 완료 불확실로 표시했다. 실습 Batch 2(2018-02-20)는 연구진 공개 로더(2017-06-30)와 달라 병합·제외 인덱스를 그대로 적용하지 않았다.')])
 page('q1','Q1. 수명 분포와 배치 간 차이',select('q1',height=165))
 page('q1_outlier','Q1. 짧은 셀의 원인 가설과 이상치 검토',select('q1_outlier'))
 page('q2_knee','Q2. 용량 열화와 가속 열화 시작점',
  select('q2',[],height=165)+select('q2_knee',[],height=120)+[
  para('발견 → 초기 피처','완만한 감소 뒤 말기 가속이 보이지만 초기 기울기가 양수인 셀도 12/23/1개다. 기울기와 수명 상관은 +0.578/-0.271/+0.184로 변해 추가 효과를 CV로 검증한다. 곡선 길이는 측정 종료에도 영향을 받는다.'),
  para('knee 규칙과 한계','QD 0.80~1.32 Ah, cycle≥10에 11점 중앙값 평활화 후 기록 15~85% 구간의 연속 두 직선을 탐색했다. SSE 20% 개선·후반 음의 기울기·절댓값 1.5배 초과 조건이다. 이는 물리적 확정점이 아닌 후보이며 전체 기록을 사용하므로 입력에서 제외한다. 상대 위치는 셀별 knee/life 중앙값; 결측 라벨은 제외한다. Batch 2 라벨 이후 기록도 탐색에 포함한다.')])
 page('q3','Q3. 초기 ΔQ 곡선과 장·단수명 그룹',
  select('q3',['계산 정의'],height=155)+select('q3_stats',[],images=False)+[
  para('발견과 그룹 해석','선은 그룹 중앙값, 음영은 IQR이다. Batch 2 단수명은 약 3 V에서 더 깊은 음의 변화와 큰 log 분산을 보인다. Batch 1·3에는 <500 그룹이 없어 중간/장수명을 비교했다. Batch 2 장수명 n=3은 안정성이 제한적이다.'),
  para('피처 정의 → 선택','log_var_delta_q=log10(var(ΔQ, ddof=1)); 최솟값·평균도 추출했다. 동일 실제 Vdlin 3.5→2.0 V, 유효점 ≥900 조건을 사용했다. 장수명일수록 변화가 작은 경향이 일관되어 log 분산을 핵심으로 선택한다. 완료 불확실을 제외한 Batch 1에서도 ρ=-0.826(n=36)이다.')])
 page('q4_policy','Q4. 충전 정책과 수명: 고속이면 짧은가?',select('q4_policy'))
 page('q4_current','Q4. 실제 전류 패턴과 초기 열화의 관계',
  select('q4_profiles',[],tables=False,height=180)+select('q4_current',[],images=False)+[
  para('파형의 의미','cycle 5의 경과시간을 0~1로 정규화했다. 양수는 충전, 음수는 방전이다. 배치별 완료 불확실이 없는 최소·최대 수명 셀 6개 예시로, 그룹 대표 표본은 아니다. b3c28은 첫 단계 3.7C지만 뒤 단계 5.9C로 최대 I=5.90 A; 장수명 b3c38은 최대 5.00 A다.'),
  para('전류 요약과 배치 비교','초기 1~5사이클에서 I>0.1 A 중앙값의 셀별 평균을 사용했다. 전류-기울기 표본은 46/47/46, 수명 상관은 46/39/44개다. 전류-기울기와 충전시간-수명은 배치별 부호가 달라 고속 충전의 인과 효과를 확정하지 않는다.'),
  para('시사점 → 검증','c1만으로 후반 세기·전환 SOC를 표현하기 어렵다. ΔQ 기본형에 충전시간·c1을 단계적으로 추가한다. 전류 상위 분위수·단계별 체류시간은 후속 후보이며 아직 전체 셀에서 검증한 입력이 아니다.')])
 page('q5','Q5. 수명 관련 신호와 변수 중복',
  select('q5',[],height=120)+select('q5_redundancy',[],images=False)+[
  para('핵심 발견 → 처리','제시한 초기 후보 중 log 분산의 절대 상관이 세 배치에서 가장 크다. 반면 충전시간·기울기·온도는 조건 의존적이다. ΔQ 요약량 |ρ|≈0.95~0.99와 온도 중복 때문에 요약량 하나·온도 하나로 시작하고 Ridge/ElasticNet을 비교한다.'),
  para('상관의 한계','표는 쌍별 결측 제외 Spearman과 n이다. 상관은 성능·인과·독립 효과를 입증하지 않는다. 단조 중복은 VIF와 같지 않으며 필요 시 학습 fold 안에서 Pearson/VIF를 확인한다.')])
 page('features','Feature Engineering: 핵심·추가·제외 변수',select('features',['선별 절차','예측 시점과 결측']))
 page('task','회귀 선택과 Target Variable',select('task'))
 page('models','EDA 특성에 근거한 후보 모델',select('models',['왜 딥러닝·1000차원 곡선 모델을 우선 쓰지 않는가?','최종 선택 원칙']))
 page('validation','검증 전략·분석 한계·참고 자료',select('validation')+[
  para('업무 활용과 완료 범위','초기 총수명 예측은 셀 선별·교체 우선순위의 참고 근거다. 실제 ESS 적용에는 운전 조건·캘린더 열화·불확실성 검증이 필요하다. DAY 1 EDA·설계를 완료했으며 모델 학습은 DAY 2다. 9.1% MAPE는 달성 성능이 아니다.'),
  para('재현 자료와 출처','전체 해석·통계는 상세 원고와 실행 노트북에 보존했다. 실습: actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb\n데이터: kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle\n로더: github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation\nSeverson et al. (2019), Nature Energy. DOI: 10.1038/s41560-019-0356-8')])
 return pages

def compact_manuscript(c):
 pages=base_compact_manuscript(c);root=c['root']
 st=pd.read_csv(root/'results/knee_stability_summary.csv')
 cor=pd.read_csv(root/'results/current_pattern_correlations.csv')
 plan=json.loads((root/'data/processed/modeling_plan.json').read_text())
 def para(label,text):return ('p',label,text)
 def rho(feature,target,batch):return cor[(cor.feature==feature)&(cor.target==target)&(cor.batch==batch)].spearman.iloc[0]
 pages[0]['blocks'].insert(0,para('제출자','울산캠퍼스 4반 | 윤도균'))
 pages[3]['blocks']=[
 ('i','02_degradation.png',165),
 para('열화 곡선의 발견','완만한 감소 뒤 말기 가속이 보이지만 초기 10~100사이클의 기울기가 양수인 셀도 12/23/1개다. 기울기와 수명 상관은 +0.578/-0.271/+0.184다. 초기 기울기는 안정화·측정 조건의 영향도 받아 핵심 ΔQ에 추가 효과를 검증한다.'),
 ('i','15_knee_sensitivity.png',125),
 ('t',['배치','기본 후보 / 전체','knee 중앙값','27설정 모두 탐지','위치 최대 변동 / 기록'],[[b,f'{int(c["knee"].set_index("batch").loc[b,"candidates"])} / {int(c["stats"].set_index("batch").loc[b,"n"])}',f'{c["knee"].set_index("batch").loc[b,"knee_cycle_median"]:.1f}',f'{int(st.set_index("batch").loc[b,"all_settings_detected"])}',f'{st.set_index("batch").loc[b,"max_position_span"]:.1%}'] for b in GROUPS],[65,110,95,110,110]),
 para('민감도 검증 방법','기존 연속 두 직선 탐지에서 평활화 7/11/21점 × SSE 개선율 10/20/30% × 후반/전반 기울기 비율 1.2/1.5/2.0의 27설정을 비교했다. QD 0.80~1.32 Ah, cycle≥10, 기록 15~85% 탐색과 후반 음의 기울기는 동일하다.'),
 para('발견 → 모델 전략','131/139셀은 모든 설정에서 탐지됐다. 허용 설정 사이 위치 범위의 배치별 중앙값은 0%, 최대는 1%/1%/0%다. 다만 탐색 격자 자체가 기록 길이의 1%여서 작은 변화는 분해하지 못한다. 이 범위의 안정성은 물리적 knee 입증이 아니다. 전체 기록·Batch 2 라벨 이후 기록을 쓰므로 knee는 X에서 제외한다.')]
 pages[6]['blocks']=[
 ('i','14_current_pattern_population.png',180),
 ('t',['초기 전류 피처 / 수명 ρ','Batch 1 (n=46)','Batch 2 (n=39)','Batch 3 (n=44)'],[[k,*[f'{rho(k,"cycle_life",b):+.3f}' for b in GROUPS]] for k in ['charge_mean_a','charge_rms_a','charge_p95_a','charge_fraction_gt4a']],[190,100,100,100]),
 para('정의와 추출 품질','Batch 1의 cycle 1에는 유효 충전 구간이 없어 배치 간 같은 관측 창인 실제 cycle 2~5의 4개 곡선을 139셀 모두에서 사용했다. Δt>0이고 양 끝 I>0.1 A인 구간에 중간 전류와 Δt 가중치를 적용했다. 사이클별 시간 가중 평균·RMS·95분위·I>4 A 시간 비율을 계산한 뒤 4사이클 평균했다. A는 C-rate와 다르며 4 A는 탐색 기준이다.'),
 para('발견·열화와 연결',f'RMS-수명 ρ는 {rho("charge_rms_a","cycle_life","Batch 1"):+.3f}/{rho("charge_rms_a","cycle_life","Batch 2"):+.3f}/{rho("charge_rms_a","cycle_life","Batch 3"):+.3f}로 모두 음수지만 강도가 다르다. RMS-초기 QD 기울기도 {rho("charge_rms_a","qd_slope_10_100","Batch 1"):+.3f}/{rho("charge_rms_a","qd_slope_10_100","Batch 2"):+.3f}/{rho("charge_rms_a","qd_slope_10_100","Batch 3"):+.3f}(n=46/47/46)다. 반면 >4 A 비율은 수명 관계의 부호가 바뀐다. 기존 샘플 중앙값과 시간 가중 통계는 다른 요약량이며 단순한 고속충전 인과 설명은 불충분하다.'),
 para('시사점 → 후보 선별','G0 핵심 ΔQ에 RMS를 하나 추가한 G4를 기존 G1~G3와 별도로 비교한다. 평균/RMS/95분위는 서로 중복될 수 있어 동시에 확장하지 않는다. 최종 피처는 Batch 1 학습 CV로 선택하며 외부 상관으로 튜닝하지 않는다. 파형 6개 예시는 노트북에 유지했다.')]
 pages[8]['blocks'].append(para('G4 충전 패턴의 별도 비교',f'G0 + charge_rms_a를 별도로 비교한다. RMS와 ΔQ log 분산의 상관은 {rho("charge_rms_a","log_var_delta_q","Batch 1"):+.3f}/{rho("charge_rms_a","log_var_delta_q","Batch 2"):+.3f}/{rho("charge_rms_a","log_var_delta_q","Batch 3"):+.3f}(n=46/47/46)로 일부 신호가 중복된다. 따라서 추가 예측력은 CV로 검증한다. strategy_checks.model_inputs는 초기 피처 화이트리스트를 강제해 타깃·전체 knee·종료값·식별자를 차단한다.'))
 folds=pd.read_csv(root/'results/cv_split_plan.csv')
 pages[11]['blocks']=[
 para('고정한 라벨 정책과 모집단','유효 cycle_life와 종료 유효 QD≤0.885 Ah를 주 분석의 공통 기준으로 고정했다. Batch 1은 36셀·20정책 그룹, Batch 2는 39셀, Batch 3는 44셀이다. 제외 셀은 우측 검열 가능성이 있으므로 수명 결측 대체를 하지 않는다. Batch 1 전체 46셀의 민감도 분석은 주 결과와 별도로 보고하고 유리한 결과로 정책을 바꾸지 않는다.'),
 ('t',['구분','셀 수','충전 정책 그룹','역할'],[['Train / CV',plan['train_n'],plan['train_groups'],'후보·피처·파라미터 선택'],['Valid / Hold-out',plan['holdout_n'],plan['holdout_groups'],'선택 후 1회 내부 평가'],['Test / Batch 2',plan['test2_n'],9,'필수 외부 평가'],['Batch 3',plan['optional_test3_n'],8,'선택 추가 평가']],[120,60,120,190]),
 para('실제 분할 가능성 확인',f'c1·전환 SOC·c2 수치 조합으로 정책을 묶어 newstructure 접미사 차이를 같은 그룹으로 처리했다. GroupShuffleSplit(test_size=0.2, seed=42)로 29/7셀, 16/4그룹을 고정했다. Train에서 GroupKFold 5개 검증 fold는 각각 {"/".join(str(int(x)) for x in folds.valid_cells)}셀이다. Hold-out/CV 모두 정책 그룹 중복 0이며 셀별 배정 CSV와 fold 통계를 저장했다.'),
 para('Pipeline·선택 기준','각 학습 fold만으로 결측 중앙값·선형 모델 표준화·피처 선택·튜닝을 수행한다. G0~G3 및 별도 G4, 원 타깃/log 타깃을 제한된 후보로 비교한다. CV 평균 MAPE와 fold 표준편차를 함께 보고하며 유사하면 더 단순한 모델을 택한다. Hold-out은 선택에 사용하지 않는다.'),
 para('지표와 오차 분석','CV 평균·표준편차, Hold-out, Batch 2를 분리하고 MAPE·MAE·RMSE를 원 사이클 단위로 평가한다. log 예측은 exp로 복원한다. Gap=Valid−CV/Test−Valid/Test−9.1%(%p)로 정의한다. 학습 수명 범위 밖 구간·충전 정책·결측별 잔차를 점검한다. 외부 오차를 보고 재튜닝하지 않는다.'),
 para('한계와 주장 범위','Hold-out 7셀·fold 5~6셀이라 평가 변동이 클 수 있다. 종료 기준으로 라벨 진실성이 확정되는 것은 아니며 모집단 선택 편향도 남는다. 세 배치 EDA·Batch 1 전체 라벨을 이미 보아 완전한 눈가림 검증은 아니다. 모델 학습은 DAY 2이며 9.1%는 달성 성능이 아니다. 실험실 셀에서 실제 ESS로 적용하려면 추가 운전 조건·불확실성 검증이 필요하다.'),
 para('자료·출처','재현 코드: src/strategy_checks.py, 실행 노트북, data/processed, results. 실습: actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb\n데이터: kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle\n로더: github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation\nSeverson et al. (2019), Nature Energy. DOI: 10.1038/s41560-019-0356-8')]
 return pages

def build_detailed_report(root):
 root=Path(root)
 import strategy_checks
 strategy_checks.run(root)
 c=analyze(root);pages=compact_manuscript(c)
 pdfmetrics.registerFont(TTFont('ESSKorean',FONT))
 styles={k:ParagraphStyle(k,fontName='ESSKorean',fontSize=size,leading=lead,textColor=colors.HexColor("#"+color),spaceAfter=after,wordWrap='CJK',keepWithNext=k in ['title','label']) for k,size,lead,color,after in [('title',18,25,'15304b',12),('label',11,16,'137d80',5),('body',9.5,14.5,'15304b',9),('small',8.2,12,'15304b',0)]}
 def p(text,style):return Paragraph(html.escape(str(text)).replace('\n','<br/>'),styles[style])
 story=[]
 for i,page in enumerate(pages):
  if i:story.append(PageBreak())
  story.append(p(page['title'],'title'))
  for block in page['blocks']:
   if block[0]=='p':story.extend([p(block[1],'label'),p(block[2],'body')])
   elif block[0]=='i':
    from PIL import Image as PILImage
    path=root/'results'/block[1];w,h=PILImage.open(path).size;scale=min(490/w,block[2]/h)
    story.append(Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER'))
   else:
    rows=[block[1]]+block[2];t=Table([[p(v,'small') for v in row] for row in rows],colWidths=block[3] or [490/len(block[1])]*len(block[1]),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#15304b')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d8e2e8')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('SPACEAFTER',(0,0),(-1,-1),9)]));story.append(t)
 def footer(canvas,doc):
  canvas.setFont('ESSKorean',8);canvas.setFillColor(colors.HexColor('#526273'));canvas.drawString(42,25,'SKALA | ESS 배터리 수명 예측 | DAY 1 모델 전략');canvas.drawRightString(A4[0]-42,25,str(doc.page));canvas.setStrokeColor(colors.HexColor('#d8e2e8'));canvas.line(42,39,A4[0]-42,39)
 path=root/'report/DAY1_모델설계.pdf'
 SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=52,title='ESS 배터리 수명 예측 - DAY1 모델 전략',author='SKALA').build(story,onFirstPage=footer,onLaterPages=footer)
 (root/'report/DAY1_모델전략_압축.md').write_text(markdown(pages))
 return path

def enhance_notebook(root):
 import nbformat
 root=Path(root);c=analyze(root);pages=manuscript(c);upgraded=compact_manuscript(c)
 for key,idx in [('q2_knee',3),('q4_current',6),('validation',11)]:
  for page in pages:
   if page['key']==key:page['blocks']=upgraded[idx]['blocks']
 for page in pages:
  if page['key']=='features':page['blocks']=upgraded[8]['blocks']
  if page['key']=='q4_profiles':page['blocks']=[b for b in page['blocks'] if not (b[0]=='p' and b[1]=='시사점 → 패턴 피처')]
 (root/'report/DAY1_모델전략_상세.md').write_text(markdown(pages))
 lookup={p['key']:p for p in pages};path=root/'notebooks/01_EDA.ipynb';nb=nbformat.read(path,4)
 def text(keys):return markdown([lookup[k] for k in keys],images=False)
 for cell in nb.cells:
  if cell.cell_type=='markdown':
   s=cell.source
   if s.startswith('# ESS 배터리 수명 예측'):
    cell.source='# ESS 배터리 수명 예측 - DAY 1 EDA\n\n'+text(['overview'])
   elif s.startswith('**품질 기준:**') or s.startswith('## 입력 데이터·수집 구조'):
    cell.source=text(['data'])
   elif (s.startswith('- Batch 1:') or s.startswith('## Q1. Cycle Life')):cell.source=text(['q1','q1_outlier'])
   elif (s.startswith('knee는 ') or s.startswith('## Q2. 방전용량')):cell.source=text(['q2','q2_knee'])
   elif (s.startswith('**전략:** 작은 셀') or s.startswith('## Q3. ΔQ')):cell.source=text(['q3','q3_stats'])
   elif (s.startswith('**전략:** 정책 문자열') or s.startswith('## Q4. 충전 프로토콜')):cell.source=text(['q4_policy','q4_profiles','q4_current'])
   elif (s.startswith('Batch 1의 강한 초기 신호:') or s.startswith('## Q5. 어떤 초기')):cell.source=text(['q5','q5_redundancy'])
   elif (s.startswith('## 6. 회귀 모델 설계 전략') or s.startswith('## Feature Engineering:')):cell.source=text(['features','task','models','validation','mapping'])
  elif cell.cell_type=='code':
   if 'eda.tables(cells, ROOT / "results")' in cell.source and 'detailed_day1.analyze' not in cell.source:
    cell.source=cell.source.replace('eda.tables(cells, ROOT / "results")','eda.tables(cells, ROOT / "results")\nimport detailed_day1\n_ = detailed_day1.analyze(ROOT)')
   if 'results/10_charging_measurements.png' in cell.source and '13_current_profiles.png' not in cell.source:
    cell.source+='\ndisplay(Image(filename=str(ROOT / "results/13_current_profiles.png")))'
   if 'results/08_delta_examples.png' in cell.source and '11_delta_life_groups.png' not in cell.source:
    cell.source+='\ndisplay(Image(filename=str(ROOT / "results/11_delta_life_groups.png")))'
 # Add target transformation output before reproducibility section, once.
 if not any('12_target_transformation.png' in c.source for c in nb.cells if c.cell_type=='code'):
  j=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 7. 재현 환경'))
  nb.cells.insert(j,nbformat.v4.new_code_cell('display(Image(filename=str(ROOT / "results/12_target_transformation.png")))'))
 if not any('strategy_checks.run(ROOT)' in x.source for x in nb.cells if x.cell_type=='code'):
  j=next(i for i,x in enumerate(nb.cells) if x.cell_type=='markdown' and x.source.startswith('## 7. 재현 환경'))
  nb.cells[j:j]=[nbformat.v4.new_markdown_cell('## 추가 검증: Knee 민감도·전체 셀 전류·정책 그룹 분할\n\n초기 전류는 저장된 피처를 사용하며 원본부터 다시 추출하려면 `python src/strategy_checks.py --raw-dir data/raw`를 실행합니다. 모델은 학습하지 않습니다.'),nbformat.v4.new_code_cell('import strategy_checks\nchecks = strategy_checks.run(ROOT)\ndisplay(checks["stability"])\ndisplay(checks["correlations"])\ndisplay(pd.DataFrame([checks["plan"]]))\ndisplay(Image(filename=str(ROOT / "results/14_current_pattern_population.png")))\ndisplay(Image(filename=str(ROOT / "results/15_knee_sensitivity.png")))')]
 nbformat.write(nb,path)
 return path

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=p.parse_args()
 print(build_detailed_report(args.root));print(enhance_notebook(args.root))
