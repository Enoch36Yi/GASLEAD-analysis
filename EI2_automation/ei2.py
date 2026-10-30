#!/usr/bin/env python3
"""Prospective EI2 analysis support. Reads raw files; writes only a new output directory.

No acquisition, feature search, automatic exclusions, claim-register changes or manuscript prose.
Use the completed input sheet to populate recording_manifest.csv before analyzing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Task-local dependencies only. On another machine install requirements.txt normally.
for extra in (HERE/'.deps', ROOT/'tmp/ei2_preparation_2026-10-14/dependencies',
              ROOT/'tmp/draft_revision_2026-10-12/dependencies',ROOT/'.runtime'):
    if extra.is_dir():
        sys.path.append(str(extra))
import numpy as np
import h5py
import scipy
import frozen_features as frozen

ACQUIRED_CHANNELS = ('piezo','mems','ae','contact_mic')
# October 15 pre-recording protocol amendment: ai3 is physically disconnected.
CHANNELS = ('piezo','mems','ae')
FEATURES = {'piezo':'band_fraction_002000_004000__median',
            'mems':'band_fraction_700000_nyquist__median',
            'ae':'band_fraction_001000_002000__median'}
MANIFEST_FIELDS = ['slot','cycle','position','state','relative_path','use','exclusion_reason',
                   'gauge_value','gauge_units','notes']
LEAK_FIRST = {'A':{1,7,8,9,10}, 'B':{2,5,6,8,10,12}}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # Convert non-finite values to JSON null, never nonstandard NaN literals.
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
        if isinstance(v,(tuple,list)):return [clean(a) for a in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def write_csv(path, rows, fields=None):
    rows=list(rows)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def fit_rule(x, y, direction=None):
    """Frozen midpoint/boundary/tie rules; optionally restrict the fitted direction."""
    x=np.asarray(x,float);y=np.asarray(y,int)
    if len(x)!=len(y) or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid threshold inputs')
    if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:raise ValueError('Both binary classes required')
    if direction not in (None,'higher','lower'):raise ValueError('Invalid direction')
    order=np.argsort(x,kind='mergesort');xs=x[order];ys=y[order]
    split=np.concatenate(([0],np.flatnonzero(xs[1:]!=xs[:-1])+1,[len(x)])).astype(int)
    cp=np.concatenate(([0],np.cumsum(ys==1)));cn=np.concatenate(([0],np.cumsum(ys==0)))
    pt=int(np.sum(ys==1));nt=int(np.sum(ys==0))
    higher=.5*((pt-cp[split])/pt+cn[split]/nt)
    lower=.5*(cp[split]/pt+(nt-cn[split])/nt)
    thresholds=np.array([np.nextafter(xs[0],-np.inf) if k==0 else
                         np.nextafter(xs[-1],np.inf) if k==len(x) else
                         xs[k-1]+(xs[k]-xs[k-1])/2 for k in split])
    hi=int(np.argmax(higher));lo=int(np.argmax(lower))
    chosen=direction or ('higher' if higher[hi]>=lower[lo]-1e-12 else 'lower')
    i=hi if chosen=='higher' else lo;s=higher if chosen=='higher' else lower
    return {'threshold':float(thresholds[i]),'direction':chosen,'balanced_accuracy':float(s[i])}


def predict(x,rule):
    x=np.asarray(x,float)
    return (x>rule['threshold'] if rule['direction']=='higher' else x<rule['threshold']).astype(int)


def score(y,p):
    y=np.asarray(y,int);p=np.asarray(p,int)
    if len(y)!=len(p) or not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():raise ValueError('Invalid prediction inputs')
    tp=int(np.sum((y==1)&(p==1)));fn=int(np.sum((y==1)&(p==0)))
    tn=int(np.sum((y==0)&(p==0)));fp=int(np.sum((y==0)&(p==1)))
    return {'ba':float(.5*(tp/(tp+fn)+tn/(tn+fp))),'tp':tp,'fn':fn,'tn':tn,'fp':fp,'n_files':len(y)}


def loco(x,y,cycles,direction=None):
    x=np.asarray(x,float);y=np.asarray(y,int);cycles=np.asarray(cycles,int)
    out=[None]*len(x)
    for cycle in sorted(set(cycles)):
        train=cycles!=cycle;held=np.flatnonzero(~train)
        if len(held)!=2 or sorted(y[held].tolist())!=[0,1]:raise ValueError('Each cycle needs one open and one sealed file')
        rule=fit_rule(x[train],y[train],direction)
        for i in held:
            out[i]={'index':int(i),'cycle':int(cycle),'truth':int(y[i]),
                    'prediction':int(predict([x[i]],rule)[0]),'threshold':rule['threshold'],
                    'direction':rule['direction'],'n_training_files':int(train.sum())}
    return out


def assignments(schedule):
    if schedule=='A':
        for chosen in itertools.combinations(range(10),5):
            z=-np.ones(10);z[list(chosen)]=1;yield z
    elif schedule=='B':
        for a in itertools.combinations(range(6),3):
            for b in itertools.combinations(range(6,12),3):
                z=-np.ones(12);z[list(a)+list(b)]=1;yield z
    else:raise ValueError('Unknown schedule')


def randomization_test(first_minus_second, observed_order, schedule):
    d=np.asarray(first_minus_second,float);z=np.asarray(observed_order,int)
    expected=10 if schedule=='A' else 12
    if len(d)!=expected or len(z)!=expected or not np.isfinite(d).all():raise ValueError('Incomplete/nonfinite paired data')
    if not np.isin(z,[-1,1]).all():raise ValueError('Order must be +1 for open first, -1 otherwise')
    if schedule=='A' and np.sum(z==1)!=5:raise ValueError('Schedule A must have five open-first pairs')
    if schedule=='B' and any(np.sum(z[i:i+6]==1)!=3 for i in (0,6)):raise ValueError('Schedule B requires 3/6 within each mount')
    # Fixed slot outcomes under the sharp null; vary only the permitted assignments.
    observed=float(np.sum(d*z)/len(d))
    ref=[float(np.sum(d*a)/len(d)) for a in assignments(schedule)]
    # A very small fixed numerical tolerance treats floating-point ties as ties.
    tol=64*np.finfo(float).eps*max(1.,abs(observed),max(abs(v) for v in ref))
    return {'available':True,'observed_mean_difference':observed,
            'p_one_sided_higher':sum(v>=observed-tol for v in ref)/len(ref),
            'p_one_sided_lower':sum(v<=observed+tol for v in ref)/len(ref),
            'n_assignments':len(ref),'null':'sharp null of no open/sealed configuration effect on slot outcomes; no carryover',
            'tie_tolerance':tol},ref


def randomization_from_pairs(first,second,order,schedule):
    first=np.asarray(first,float);second=np.asarray(second,float)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or np.any(first<=0) or np.any(second<=0):
        return {'available':False,'reason':'Log10 statistic undefined for nonpositive/nonfinite features; no rows removed or pseudocount added'},[]
    return randomization_test(np.log10(first)-np.log10(second),order,schedule)


def baseline(x):
    x=np.asarray(x,float)
    if not len(x) or not np.isfinite(x).all():return {'available':False,'reason':'Incomplete/nonfinite baseline'}
    q25,q75=np.quantile(x,[.25,.75]);med=float(np.median(x));lo=float(np.min(x))
    return {'available':True,'n_files':len(x),'median_ac_rms':med,
            'niqr':float((q75-q25)/med) if med>0 else None,'max_min':float(np.max(x)/lo) if lo>0 else None}


def schedule_rows(schedule):
    n=10 if schedule=='A' else 12;rows=[]
    for cycle in range(1,n+1):
        for position in (1,2):
            state='OPEN' if ((cycle in LEAK_FIRST[schedule])==(position==1)) else 'SEALED'
            rows.append(dict(zip(MANIFEST_FIELDS,[str(2*cycle-2+position),str(cycle),str(position),state,'','yes','','','',''])))
    return rows


def raw_path(root, relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if not p.is_relative_to(root) or Path(relative).is_absolute():raise ValueError(f'Raw path escapes data root: {relative}')
    if p.suffix.lower() not in ('.h5','.hdf5'):raise ValueError('Expected HDF5 file')
    return p


def validate_manifest(rows,schedule):
    selected=[r for r in rows if str(r.get('use','')).strip().lower()=='yes']
    expected=schedule_rows(schedule)
    if len(selected)!=len(expected):raise ValueError(f'{schedule} needs exactly {len(expected)} selected files')
    try:selected=sorted(selected,key=lambda r:int(r['slot']))
    except (KeyError,ValueError) as e:raise ValueError('Selected rows need numeric slot IDs') from e
    for r,e in zip(selected,expected):
        if any(str(r.get(k,'')).strip()!=e[k] for k in ('slot','cycle','position','state')):raise ValueError(f'Slot {e["slot"]}: cycle/order/state does not match fixed schedule')
        if not str(r.get('relative_path','')).strip():raise ValueError(f'Slot {e["slot"]} missing file mapping from input sheet')
        folders={part.casefold() for part in Path(r['relative_path'].replace('\\','/')).parts[:-1]}
        if ('no_leak' in folders and r['state']!='SEALED') or ('leak' in folders and r['state']!='OPEN'):
            raise ValueError(f'Slot {e["slot"]}: acquisition target folder contradicts logged state; do not silently relabel')
    paths=[r['relative_path'].strip().replace('\\','/').casefold() for r in selected]
    if len(set(paths))!=len(paths):raise ValueError('A selected raw file cannot occupy two slots')
    all_paths=[]
    for r in rows:
        if r.get('use','').strip().lower() not in ('yes','no'):raise ValueError('Every row needs use=yes or no')
        if not r.get('relative_path','').strip():raise ValueError('Every attempt needs a raw file path')
        all_paths.append(r['relative_path'].strip().replace('\\','/').casefold())
        if r['use'].strip().lower()=='no' and not r.get('exclusion_reason','').strip():raise ValueError('Excluded attempts/warmups need documented reasons')
    if len(all_paths)!=len(set(all_paths)):raise ValueError('Each raw file must have only one manifest entry')
    return selected


def inspect_h5(path,minimum_seconds=29.5):
    result={'file':str(path),'errors':[],'warnings':[],'channels':[]}
    try:
        with h5py.File(path,'r') as f:
            if 'metadata' not in f or 'raw' not in f:raise ValueError('Missing raw or metadata group')
            fs=float(f['metadata'].attrs.get('sample_rate_hz',np.nan));result['sample_rate_hz']=fs
            timestamp=f['metadata'].attrs.get('timestamp','')
            if isinstance(timestamp,bytes):timestamp=timestamp.decode('utf8')
            result['timestamp']=str(timestamp)
            result['stored_channels']=list(f['raw'].keys())
            result['excluded_channels']={'contact_mic':{'present':'contact_mic' in f['raw'],
                'reason':'Deliberately disconnected before recording; no numeric QC gate, features or statistics'}}
            if not np.isfinite(fs) or fs!=2_000_000:result['errors'].append('sample_rate_hz must be 2000000')
            counts=[]
            for c in CHANNELS:
                if c not in f['raw']:result['errors'].append(f'Missing channel {c}');continue
                ds=f['raw'][c]
                if ds.ndim!=1:result['errors'].append(f'{c} is not one-dimensional');continue
                n=int(ds.shape[0]);counts.append(n);bad=0;rail=0;lo=math.inf;hi=-math.inf
                for start in range(0,n,2_000_000):
                    x=np.asarray(ds[start:start+2_000_000],float);finite=np.isfinite(x);bad+=int((~finite).sum());a=x[finite]
                    if len(a):lo=min(lo,float(a.min()));hi=max(hi,float(a.max()));rail+=int((np.abs(a)>=4.99).sum())
                duration=n/fs if np.isfinite(fs) and fs>0 else math.nan
                if not n or not math.isfinite(duration) or duration<minimum_seconds:result['errors'].append(f'{c} short/invalid duration ({duration:g}s)')
                if bad:result['errors'].append(f'{c}: {bad} nonfinite samples')
                if rail:result['warnings'].append(f'{c}: {rail} samples with |V|>=4.99; document, do not automatically exclude')
                if math.isfinite(duration) and duration>30.5:result['warnings'].append(f'{c}: duration exceeds 30.5s; protocol deviation review')
                result['channels'].append({'channel':c,'n_samples':n,'duration_s':duration,'min_v':lo,'max_v':hi,'rail_samples':rail,'nonfinite_samples':bad})
            if len(set(counts))>1:result['errors'].append('Channel sample counts differ')
    except (OSError,ValueError,KeyError,TypeError) as e:result['errors'].append(str(e))
    return result


def extract_h5(path):
    """Read one channel/window at a time; use frozen August 6 spectral functions verbatim."""
    rows=[];windows=[]
    with h5py.File(path,'r') as f:
        fs=int(f['metadata'].attrs['sample_rate_hz'])
        if fs!=2_000_000:raise ValueError('Feature extraction requires 2 MS/s')
        counts=[len(f['raw'][c]) for c in CHANNELS]
        if len(set(counts))!=1:raise ValueError('Unequal channel lengths')
        nwin=counts[0]//fs
        if not nwin:raise ValueError('No complete one-second windows')
        for c in CHANNELS:
            values=[];ac=[];dc=[]
            for i in range(nwin):
                x=np.asarray(f['raw'][c][i*fs:(i+1)*fs],dtype=np.float64)
                if not np.isfinite(x).all():raise ValueError('Nonfinite raw data')
                _,_,s=frozen.periodogram_features(x,fs)
                value=float(s[FEATURES[c].removesuffix('__median')]);mean=float(np.mean(x));rms=float(np.sqrt(np.mean((x-mean)**2)))
                values.append(value);ac.append(rms);dc.append(mean)
                windows.append({'channel':c,'window_index':i,'feature_value':value,'ac_rms_v':rms,'dc_mean_v':mean})
            # np.median intentionally does not discard undefined feature values.
            rows.append({'channel':c,'feature':FEATURES[c],'feature_value':float(np.median(values)),
                         'median_window_ac_rms':float(np.median(ac)),'median_window_mean_v':float(np.median(dc)),
                         'n_full_windows':nwin,'omitted_terminal_samples':counts[0]-nwin*fs})
    return rows,windows


def evaluate(feature_rows,schedule,rules):
    summary={'status':'Analysis output; Enoch and team review pending; no claim eligibility inferred',
             'schedule':schedule,'channels':{},'scope':'This pipe, randomized open/apertured-pad versus sealed/solid-pad configurations; not the original four-condition endpoint'}
    summary['excluded_channels']={'contact_mic':'PVDF/ai3 deliberately disconnected by pre-recording October 15 amendment; no analysis'}
    predictions=[];pairs=[];references=[];baseline_rows=[]
    for c in CHANNELS:
        rows=sorted((r for r in feature_rows if r['channel']==c),key=lambda r:int(r['slot']))
        x=np.array([float(r['feature_value']) for r in rows]);y=np.array([r['state']=='OPEN' for r in rows],int)
        cycles=np.array([int(r['cycle']) for r in rows]);rule=rules[c]
        finite=bool(np.isfinite(x).all());ch={'feature':FEATURES[c],'locked_rule':rule,'descriptive_only':c!='piezo'}
        if finite:
            p=predict(x,rule);ch['locked_transfer']=score(y,p)
            for r,pr in zip(rows,p):predictions.append({'channel':c,'method':'locked_August6','slot':r['slot'],'cycle':r['cycle'],'truth':int(r['state']=='OPEN'),'prediction':int(pr),'threshold':rule['threshold'],'direction':rule['direction']})
            # For secondary channels use the August 6 modal-feature direction, not a new outcome-selected tail.
            fitted=loco(x,y,cycles,rule['direction']);ch['within_session_loco']=score(y,[r['prediction'] for r in fitted])
            for r,pr in zip(rows,fitted):predictions.append({'channel':c,'method':'within_session_loco','slot':r['slot'],**pr})
        else:ch['locked_transfer']=ch['within_session_loco']={'available':False,'reason':'Nonfinite feature; complete cohort score unavailable; no subset scored'}
        first=[];second=[];order=[]
        for cy in sorted(set(cycles)):
            a,b=[r for r in rows if int(r['cycle'])==cy]
            first.append(float(a['feature_value']));second.append(float(b['feature_value']));z=1 if a['state']=='OPEN' else -1;order.append(z)
            o=a if z==1 else b;n=b if z==1 else a
            logdiff=math.log10(o['feature_value'])-math.log10(n['feature_value']) if o['feature_value']>0 and n['feature_value']>0 and np.isfinite([o['feature_value'],n['feature_value']]).all() else None
            pairs.append({'channel':c,'cycle':cy,'mount':1 if schedule=='A' or cy<=6 else 2,'first_state':a['state'],
                          'open_value':o['feature_value'],'sealed_value':n['feature_value'],'log10_open_minus_sealed':logdiff})
        rand,ref=randomization_from_pairs(first,second,order,schedule)
        rand['inferential_role']='Protocol IEPE one-sided higher test' if c=='piezo' else 'Descriptive secondary diagnostic only; no multiplicity-adjusted significance claim'
        rand['predefined_direction']=rule['direction']
        if rand['available']:rand['p_in_predefined_direction']=rand['p_one_sided_higher' if rule['direction']=='higher' else 'p_one_sided_lower']
        ch['paired_randomization']=rand
        references.extend({'channel':c,'assignment':i+1,'mean_log10_difference':v} for i,v in enumerate(ref))
        ch['sealed_ac_state']=baseline([float(r['median_window_ac_rms']) for r in rows if r['state']=='SEALED'])
        baseline_rows.append({'channel':c,**ch['sealed_ac_state']})
        if c=='piezo':ch['within_1e_minus12_of_locked_threshold']=[r['slot'] for r in rows if abs(float(r['feature_value'])-rule['threshold'])<=1e-12]
        summary['channels'][c]=ch
    source=sorted((r for r in feature_rows if r['channel']=='piezo'),key=lambda r:int(r['slot']))
    y=[int(r['state']=='OPEN') for r in source];cy=[int(r['cycle']) for r in source]
    summary['controls']={}
    for name,key in [('elapsed_time','elapsed_s'),('position','position')]:
        vals=np.array([float(r.get(key,math.nan)) for r in source])
        if np.isfinite(vals).all():
            pr=loco(vals,y,cy);summary['controls'][name]=score(y,[r['prediction'] for r in pr])
            predictions.extend({'channel':'control','method':name,'slot':r['slot'],**p} for r,p in zip(source,pr))
        else:summary['controls'][name]={'available':False,'reason':'Missing/invalid timestamps or position; no ordinal-time substitution'}
    iepe=summary['channels']['piezo'];transfer=iepe['locked_transfer'];rand=iepe['paired_randomization']
    controls_ok=all('ba' in v and v['ba']<=.70 for v in summary['controls'].values())
    summary['protocol_criteria']={'locked_rule_BA_ge_0_90':transfer.get('ba',-1)>=.90,
        'paired_p_le_0_01_and_both_controls_BA_le_0_70':bool(rand.get('available') and rand['p_one_sided_higher']<=.01 and controls_ok),
        'interpretation':'Criteria met/not met only; no automatic author approval, absence-of-effect or broad transfer conclusion'}
    return summary,predictions,pairs,references,baseline_rows


def plots(out,feature_rows,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    synthetic=bool(feature_rows) and all(r.get('synthetic') is True for r in feature_rows)
    label='SYNTHETIC DEMO ONLY | ' if synthetic else ''
    fig,axes=plt.subplots(1,len(CHANNELS),figsize=(4.2*len(CHANNELS),4),squeeze=False)
    for ax,c in zip(axes.ravel(),CHANNELS):
        a=[r for r in pairs if r['channel']==c]
        for r in a:
            cy=r['cycle']
            ax.plot([cy-.12,cy+.12],[r['sealed_value'],r['open_value']],color='#777777',alpha=.65,linewidth=.8)
            ax.scatter(cy-.12,r['sealed_value'],color='#0072B2',marker='o',s=20,label='SEALED' if cy==1 else None)
            ax.scatter(cy+.12,r['open_value'],color='#D55E00',marker='^',s=22,label='OPEN' if cy==1 else None)
        ax.set_xticks([r['cycle'] for r in a]);ax.set_xlabel('Cycle');ax.set_title(c)
        if any(r['mount']==2 for r in a):
            ax.axvline(6.5,color='black',linestyle='--',linewidth=.7)
            ax.text(.02,.98,'Mount 1 | Mount 2 at dashed line',transform=ax.transAxes,va='top',fontsize=7)
        ax.set_ylabel('Spectral bandwidth (Hz)' if c=='contact_mic' else 'Recorded-voltage spectral fraction')
        ax.grid(alpha=.15)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(label+'EI2 paired observations | within-cycle links; repeated installation',fontsize=11)
    fig.tight_layout();fig.savefig(out/'paired_features.png',dpi=180);fig.savefig(out/'paired_features.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,len(CHANNELS),figsize=(4.2*len(CHANNELS),3.5),squeeze=False)
    for ax,c in zip(axes.ravel(),CHANNELS):
        a=[r for r in pairs if r['channel']==c]
        for r in a:
            ax.scatter(r['cycle'],r['log10_open_minus_sealed'] if r['log10_open_minus_sealed'] is not None else np.nan,
                       color='#0072B2' if r['mount']==1 else '#D55E00',s=22)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks([r['cycle'] for r in a])
        if any(r['mount']==2 for r in a):ax.axvline(6.5,color='black',linestyle='--',linewidth=.7)
        ax.set_xlabel('Cycle');ax.set_title(c);ax.set_ylabel('log10(OPEN) - log10(SEALED)');ax.grid(alpha=.15)
    fig.suptitle(label+'Paired feature differences; undefined values remain unplotted',fontsize=11)
    fig.tight_layout();fig.savefig(out/'paired_log10_differences.png',dpi=180);fig.savefig(out/'paired_log10_differences.pdf');plt.close(fig)
    fig,axes=plt.subplots(len(CHANNELS),2,figsize=(10,2.5*len(CHANNELS)),squeeze=False)
    for i,c in enumerate(CHANNELS):
        a=sorted([r for r in feature_rows if r['channel']==c],key=lambda r:int(r['slot']))
        for state,color,marker in [('OPEN','#D55E00','^'),('SEALED','#0072B2','o')]:
            z=[r for r in a if r['state']==state]
            for j,key in enumerate(['median_window_ac_rms','median_window_mean_v']):
                axes[i,j].scatter([int(r['slot']) for r in z],[r[key] for r in z],c=color,marker=marker,label=state,s=20)
        axes[i,0].set_ylabel(c+' AC RMS (V)');axes[i,1].set_ylabel('DC mean (V)')
        if all(r['median_window_ac_rms']>0 for r in a):axes[i,0].set_yscale('log')
        for ax in axes[i]:ax.grid(alpha=.15)
    axes[0,0].legend();axes[-1,0].set_xlabel('Recording slot');axes[-1,1].set_xlabel('Recording slot')
    fig.suptitle(label+'Recorded channel state',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'channel_state.png',dpi=180);fig.savefig(out/'channel_state.pdf');plt.close(fig)


def verify_package():
    p=HERE/'PACKAGE_LOCK.json'
    if not p.exists():raise ValueError('Package has not been frozen; PACKAGE_LOCK.json missing')
    lock=json.loads(p.read_text(encoding='utf8'))
    for name,digest in lock['files'].items():
        if sha256(HERE/name)!=digest:raise ValueError(f'Package changed after preparation: {name}; do not conceal a post-recording edit')
    return lock


def new_output(path):
    p=Path(path).resolve()
    protected=['Stage_1','Stage_2','Stage_3','Stage_4','Stage_5','release','release_archive','tables','reports','src','figures','cache','manuscript','ai_notes']
    if any(p.is_relative_to((ROOT/name).resolve()) for name in protected):
        raise ValueError('Output must be outside frozen/authoritative source folders')
    if any(part.casefold().startswith('dropbox') for part in p.parts):raise ValueError('Dropbox writes are not authorized')
    if p.exists():raise ValueError(f'Output already exists: {p}. Choose a new folder; existing results are never overwritten.')
    p.mkdir(parents=True);return p


def inventory(data_root,out):
    root=Path(data_root).resolve()
    if Path(out).resolve().is_relative_to(root):raise ValueError('Output must be outside the raw data folder')
    if not root.is_dir():raise ValueError('Data root not found')
    files=sorted({*root.rglob('*.h5'),*root.rglob('*.hdf5')})
    if not files:raise ValueError('No HDF5 files found')
    out=new_output(out);records=[]
    for p in files:
        print('Technical inspection:',p.name,flush=True)
        q=inspect_h5(p);q['relative_path']=p.relative_to(root).as_posix();q['sha256']=sha256(p);records.append(q)
    write_json(out/'technical_inventory.json',records)
    write_csv(out/'file_inventory.csv',[{'relative_path':r['relative_path'],'sha256':r['sha256'],
        'timestamp':r.get('timestamp',''),'errors':'; '.join(r['errors']),'warnings':'; '.join(r['warnings'])} for r in records])
    print('Inventory saved:',out)


def analyze(data_root,manifest,session_path,out):
    lock=verify_package();root=Path(data_root).resolve()
    source_hashes=json.loads((HERE/'SOURCE_HASHES.json').read_text(encoding='utf8'))
    for rel in ('1012_secondeitrip.md','1014_opus_EI2_analysis_prompt.md'):
        p=ROOT/rel
        if p.is_file() and sha256(p)!=source_hashes[rel]:
            raise ValueError(f'{rel} changed after package preparation; reconcile the protocol amendment and version the runner before feature computation')
    if Path(out).resolve().is_relative_to(root):raise ValueError('Output must be outside the raw data folder')
    manifest=Path(manifest).resolve();session_path=Path(session_path).resolve()
    session=json.loads(session_path.read_text(encoding='utf8'));schedule=session.get('schedule')
    if schedule not in LEAK_FIRST:raise ValueError('Session schedule must be A or B')
    if session.get('synthetic') is True:raise ValueError('Synthetic session must use demo command')
    if session.get('protocol_locked_before_first_recording') is not True:raise ValueError('Transcribe the actual pre-recording protocol-lock status from the completed sheet; do not infer it from data')
    if not session.get('completed_input_sheet_source'):raise ValueError('Identify the completed input sheet used for mapping and lock attestation')
    all_rows=read_csv(manifest);selected=validate_manifest(all_rows,schedule)
    mapped={raw_path(root,r['relative_path']) for r in all_rows}
    if any(not p.is_file() for p in mapped):raise ValueError('Manifest references a missing raw file')
    existing={p.resolve() for p in root.rglob('*') if p.suffix.lower() in ('.h5','.hdf5')}
    if existing-mapped:raise ValueError('Unlisted HDF5 files exist: list warmups and rejected attempts as use=no with reasons; do not silently omit them')
    checks=[];hashes={};problems=[];warnings=[]
    for row in all_rows:
        p=raw_path(root,row['relative_path']);digest=sha256(p);hashes[row['relative_path']]=digest
        if row['use'].strip().lower()=='yes':
            print('Technical inspection:',p.name,flush=True);q=inspect_h5(p);q['slot']=row['slot'];checks.append(q)
            problems.extend(f'{p.name}: {s}' for s in q['errors']);warnings.extend(f'{p.name}: {s}' for s in q['warnings'])
    if len(set(hashes.values()))!=len(hashes):problems.append('Identical raw-file hashes found; verify copies and slot mapping before scoring')
    # Validate timestamps before feature computation. Acquisition uses end-of-recording save timestamps.
    times=[]
    for row,q in zip(selected,sorted(checks,key=lambda q:int(q['slot']))):
        try:times.append(datetime.strptime(q.get('timestamp',''),'%Y%m%d_%H%M%S'))
        except ValueError:problems.append(f'Slot {row["slot"]}: missing/unparseable metadata timestamp')
    if not session.get('session_date'):problems.append('Session record must give the actual recording date')
    elif any(t.date().isoformat()!=session['session_date'] for t in times):problems.append('Selected file timestamps do not match session_date; reconcile before scoring')
    if len(times)==len(selected) and any(b<=a for a,b in zip(times,times[1:])):problems.append('Save timestamps are not strictly increasing in logged slot order')
    # Write a complete preflight even when scoring is blocked. Never exclude/replace files here.
    out=new_output(out)
    write_json(out/'preflight.json',{'errors':problems,'warnings':warnings,'checks':checks,'raw_hashes':hashes})
    write_csv(out/'manifest_used.csv',all_rows,MANIFEST_FIELDS)
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'package_lock':lock,'manifest_sha256':sha256(manifest),
        'session_record_sha256':sha256(session_path),'session_record':session,'raw_sha256':hashes,
        'software':{'python':platform.python_version(),'numpy':np.__version__,'h5py':h5py.__version__,'scipy':scipy.__version__},
        'source_changes_after_preparation':{}}
    for rel,digest in json.loads((HERE/'SOURCE_HASHES.json').read_text(encoding='utf8')).items():
        p=ROOT/rel
        if p.is_file() and sha256(p)!=digest:provenance['source_changes_after_preparation'][rel]='changed; archived function snapshot is still used'
    write_json(out/'provenance.json',provenance)
    if problems:raise ValueError('Preflight blocked scoring; see '+str(out/'preflight.json'))
    feature_rows=[];window_rows=[]
    cache=out/'per_file';cache.mkdir()
    for i,row in enumerate(selected):
        p=raw_path(root,row['relative_path']);print(f'Extracting {i+1}/{len(selected)}: {p.name}',flush=True)
        values,windows=extract_h5(p)
        # Detect a source modified between hashing and extraction.
        if sha256(p)!=hashes[row['relative_path']]:raise ValueError('Raw file changed during analysis: '+p.name)
        extras={**row,'elapsed_s':(times[i]-times[0]).total_seconds(),'raw_sha256':hashes[row['relative_path']]}
        enriched=[{**extras,**v} for v in values];feature_rows.extend(enriched)
        enriched_windows=[{'slot':row['slot'],'cycle':row['cycle'],'state':row['state'],**v} for v in windows];window_rows.extend(enriched_windows)
        write_json(cache/f'slot_{int(row["slot"]):02d}.json',{'feature_rows':enriched,'window_rows':enriched_windows})
    rules=json.loads((HERE/'LOCKED_RULES.json').read_text(encoding='utf8'))['rules']
    summary,predictions,pairs,reference,baselines=evaluate(feature_rows,schedule,rules)
    summary['evidence_approval_reported']=session.get('new_evidence_source_approved') is True
    summary['evidence_status']='Source approval reported; individual results still await Enoch review' if summary['evidence_approval_reported'] else 'NOT CLAIM-ELIGIBLE pending Enoch source approval'
    summary['corrected_orifice_mm']=1.89
    summary['warnings']=warnings
    write_csv(out/'file_features.csv',feature_rows);write_csv(out/'window_features.csv',window_rows)
    write_csv(out/'predictions.csv',predictions);write_csv(out/'paired_features.csv',pairs)
    write_csv(out/'randomization_reference.csv',reference);write_csv(out/'sealed_ac_state.csv',baselines)
    write_json(out/'summary.json',summary)
    try:plots(out,feature_rows,pairs)
    except ImportError as e:summary['plot_warning']=str(e);write_json(out/'summary.json',summary)
    report(out,summary,synthetic=False)
    write_json(out/'OUTPUT_HASHES.json',{p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()})
    print('Analysis saved for Opus/Enoch review:',out)


def report(out,summary,synthetic=False):
    lines=['# '+('SYNTHETIC DEMO ONLY' if synthetic else 'EI2 analysis for review'),'','No manuscript or claim-register changes. Human and team review pending.','',
           '| Channel | Locked-rule BA | Within-session LOCO BA | Paired p in preset direction |','|---|---:|---:|---:|']
    for c,v in summary['channels'].items():
        lines.append(f'| {c} | {v["locked_transfer"].get("ba","unavailable")} | {v["within_session_loco"].get("ba","unavailable")} | {v["paired_randomization"].get("p_in_predefined_direction","unavailable")} |')
    lines+=['','Only the IEPE higher-direction test is the protocol primary randomization test. Other channel rules/tails are descriptive and require Opus to check the documented secondary-direction interpretation.',
            '','Controls: '+json.dumps(summary['controls']), '', 'Criteria: '+json.dumps(summary['protocol_criteria']), '',
            'The randomized intervention changes the pad configuration with the air escape state. It does not establish a leak-only mechanism, original size boundary, independent installation replication or field transfer.',
            'The test conditions on the stated randomization and sharp null, including no consequential carryover. Meeting the protocol control cutoff does not prove absence of drift or position effects. In Schedule A, the balanced allocation can make the fitted position control score BA 0 through reversal of the training majority; do not describe that as empirical proof of chance-level behavior.',
            'Gauge values and notes remain in manifest_used.csv. Missing gauge readings remain missing; no pressure value is inferred.',
            'PVDF/contact_mic/ai3 was deliberately disconnected by the October 15 pre-recording amendment and is excluded from numeric QC and every analysis. This does not alter the August 6 study.',
            'Review preflight/provenance, per-file values, failures/exclusions, directions and exact tails before interpreting criteria. A nonsignificant test does not prove no effect.']
    if not synthetic:lines.insert(4,summary['evidence_status'])
    (out/'ANALYSIS_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def demo(out):
    out=new_output(out);rules=json.loads((HERE/'LOCKED_RULES.json').read_text(encoding='utf8'))['rules'];features=[]
    for r in schedule_rows('A'):
        y=r['state']=='OPEN';cy=int(r['cycle'])
        for c in CHANNELS:
            value=(.08 if y else .01) if c!='ae' else (.0005 if y else .002)
            if c=='contact_mic':value=80000 if y else 40000
            features.append({**r,'channel':c,'feature_value':value,'median_window_ac_rms':.01+cy*.0001,
                'median_window_mean_v':0.,'elapsed_s':int(r['slot'])*100,'synthetic':True})
    summary,predictions,pairs,ref,baselines=evaluate(features,'A',rules);summary['synthetic']=True
    write_json(out/'summary.json',summary);write_csv(out/'file_features.csv',features);write_csv(out/'predictions.csv',predictions)
    write_csv(out/'paired_features.csv',pairs);write_csv(out/'randomization_reference.csv',ref);write_csv(out/'sealed_ac_state.csv',baselines)
    try:plots(out,features,pairs)
    except ImportError as e:summary['plot_warning']=str(e);write_json(out/'summary.json',summary)
    report(out,summary,True)
    write_json(out/'OUTPUT_HASHES.json',{p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()})
    print('Synthetic demo only:',out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('inventory');q.add_argument('--data-root',required=True);q.add_argument('--out',required=True)
    q=s.add_parser('analyze');q.add_argument('--data-root',required=True);q.add_argument('--manifest',required=True)
    q.add_argument('--session',required=True);q.add_argument('--out',required=True)
    q=s.add_parser('demo');q.add_argument('--out',required=True)
    q=s.add_parser('prepare');q.add_argument('--out',required=True)
    s.add_parser('verify-package')
    a=p.parse_args(argv)
    try:
        if a.command=='inventory':inventory(a.data_root,a.out)
        elif a.command=='analyze':analyze(a.data_root,a.manifest,a.session,a.out)
        elif a.command=='demo':demo(a.out)
        elif a.command=='prepare':
            target=new_output(a.out)
            write_csv(target/'recording_manifest.csv',schedule_rows('A'),MANIFEST_FIELDS)
            (target/'session_record.json').write_bytes((HERE/'SESSION_RECORD.template.json').read_bytes())
            print('Working copies prepared for transcription from the completed input sheet:',target)
        else:print('Package checksums match:',verify_package()['prepared_utc'])
    except (ValueError,OSError,KeyError) as e:
        print('STOP:',e,file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
