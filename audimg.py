"""
audimg.py
Decoding Model for Heard / Imagined Pitch and Timbre
Michael A. Casey, Dartmouth College, Aug-Dec 2019

Library for auditory imagery fMRI experiment
Data preprocessing, classifier training, result grouping, result plotting
"""

#import matplotlib.pyplot as pl
import mvpa2.suite as P
import numpy as np
import bids.grabbids as gb
from os.path import join as opj
import csv
import glob
import pickle
from scipy.stats import wilcoxon
import sys
#import pdb

pl = P.pl # convenience for access to plotting functions

# data = 3mmx3mmx3mm

ROOTDIR='/isi/music/auditoryimagery2'
DATADIR=opj(ROOTDIR, 'am2/data/fmriprep/fmriprep/')
OUTDIR=opj(ROOTDIR, 'results_audimg_subj_task')

MRISPACE= 'MNI152NLin2009cAsym' # if using fmriprep_2mm then MRISPACE='MNI152NLin6Asym' 
PARCELLATION='desc-aparcaseg_dseg'# per-subject ROI parcellation in MRISPACE

# List of tasks to evaluate
tasks=['pch-height','pch-class','pch-hilo','timbre']

subjects = ['sid001401', 'sid000388', 'sid001415', 'sid001419', 'sid001410', # Batch 1 (June-July 2019)
            'sid001541', 'sid001427','sid001088','sid001564','sid001581','sid001594'] # Batch 2 (Oct-Nov 2019)

accessions = { # Map subject ids to assession numbers
    # Batch 1, subjects 1-5
    'sid001401':'A002636', 
    'sid000388':'A002655', 
    'sid001415':'A002652', 
    'sid001419':'A002659', 
    'sid001410':'A002677',
    # Batch 2, subjects 6-11
    'sid001541':'A002979',
    'sid001427':'A002996',
    'sid001088':'A003000',
    'sid001564':'A003037',
    'sid001581':'A003067',
    'sid001594':'A003098'
}

tonalities={ # keys
    # Batch 1, subjects 1-5
    'sid001401':'E', 
    'sid000388':'E', 
    'sid001415':'E', 
    'sid001419':'E', 
    'sid001410':'E',
    # Batch 2, subjects 6-11
    'sid001541':'F',
    'sid001427':'E',
    'sid001088':'F',
    'sid001564':'E',
    'sid001581':'F',
    'sid001594':'E'    
}

# Read run_legend to re-order runs by task
# We could hard-code this as a dict, only need the first row for run-order 
with open(opj(ROOTDIR,'targets/run_legend.csv')) as csvfile:
    legend={}
    for subj, acc in accessions.items():
        legend[acc]=[]
    reader = csv.reader(csvfile, delimiter=',')    
    for row in reader:
        legend[row[0]].append(row[2])

# ROI keys->labels map (for figures / reporting)
# Dependency: PARCELLATION
roi_map={
1000:    "ctx-lh-unknown",        #190519
1001:    "ctx-lh-bankssts",       #196428
1002:    "ctx-lh-caudalanteriorcingulate", #7d64a0
1003:    "ctx-lh-caudalmiddlefrontal",    #641900
1004:    "ctx-lh-corpuscallosum", #784632
1005:    "ctx-lh-cuneus", #dc1464
1006:    "ctx-lh-entorhinal",     #dc140a
1007:   "ctx-lh-fusiform",       #b4dc8c
1008:    "ctx-lh-inferiorparietal",       #dc3cdc
1009:    "ctx-lh-inferiortemporal",       #b42878
1010:    "ctx-lh-isthmuscingulate",       #8c148c
1011:    "ctx-lh-lateraloccipital",       #141e8c
1012:    "ctx-lh-lateralorbitofrontal",   #234b32
1013:    "ctx-lh-lingual",        #e18c8c
1014:    "ctx-lh-medialorbitofrontal",    #c8234b
1015:    "ctx-lh-middletemporal", #a06432
1016:    "ctx-lh-parahippocampal",        #14dc3c
1017:    "ctx-lh-paracentral",    #3cdc3c
1018:    "ctx-lh-parsopercularis",        #dcb48c
1019:    "ctx-lh-parsorbitalis",  #146432
1020:    "ctx-lh-parstriangularis",       #dc3c14
1021:    "ctx-lh-pericalcarine",  #78643c
1022:    "ctx-lh-postcentral",    #dc1414
1023:    "ctx-lh-posteriorcingulate",     #dcb4dc
1024:    "ctx-lh-precentral",     #3c14dc
1025:    "ctx-lh-precuneus",      #a08cb4
1026:    "ctx-lh-rostralanteriorcingulate",       #50148c
1027:    "ctx-lh-rostralmiddlefrontal",   #4b327d
1028:    "ctx-lh-superiorfrontal",        #14dca0
1029:    "ctx-lh-superiorparietal",       #14b48c
1030:    "ctx-lh-superiortemporal",       #8cdcdc
1031:    "ctx-lh-supramarginal",  #50a014
1032:    "ctx-lh-frontalpole",    #640064
1033:    "ctx-lh-temporalpole",   #464646
1034:    "ctx-lh-transversetemporal",     #9696c8
1035:    "ctx-lh-insula" #ffc020
}

# Generate RH cortical map
for k in roi_map.keys():
    roi_map[k+1000]=roi_map[k].replace('lh','rh')

def get_subject_mask(subject, run=1, rois=[1030,2030], path=DATADIR, 
                     space=MRISPACE,
                     parcellation=PARCELLATION):
    """
    Get subject mask by run and ROI key to apply to a dataset
    (rois are in DATADIR/PARCELLATION.tsv)

    inputs:
        subject  - sid00[0-9]{4}
        run      - which run to use for parcellation (redundant?) [1-8]
        rois     - list of regions of interest for mask [1030,2030]
        path     - dir containing roi parcellations [DATADIR]
       space     - parcellation space [MRISPACE]
     parcellation- file  [PARCELLATION]

    outputs:
        mask_ds  - pymvpa Dataset containing mask data {0,[rois]}
    """
    fname = opj(path, subject, 'func', '%s_task-*_run-%02d_space-%s_%s.nii.gz'%(subject,run,space, parcellation))
    #print fname
    fname = glob.glob(fname)[0]
    ds=P.fmri_dataset(fname)
    found=np.where(np.isin(ds.samples,rois))[1]
    return ds[:,found]

def get_subject_ds(subject, cache=True, cache_dir='ds_cache'):
    """Assemble pre-processed datasets    
    load subject original data (no mask applied)
    optionally cache for faster loading during model training/testing
    purpose: re-use unmasked dataset, applying mask when necessary    

    inputs:
        subject  - sid00[0-9]{4}    
        cache    - whether to use cached datasets [True]
     cache_dir   - where to store / load cached datasets ['ds_cache']

    outputs:
        data     - subject original data (no mask applied)
    """
    swap_runs=[2,1,4,3,6,5,8,7]
    layout = gb.BIDSLayout(DATADIR)
    ext = 'desc-preproc_bold.nii.gz'
    cache_filename = '%s/%s.ds_cache.nii.gz'%(cache_dir, subject)
    cache_fail=False
    if cache:
        try:
            data=P.Dataset.from_hdf5(cache_filename)        
        except:
            cache_fail=True
    if not cache or cache_fail:
        data=[]
        for run in range(1,9):
            r=run if legend[accessions[subject]][0]=='HT' else swap_runs[run-1]
            f=layout.get(subject=subject, extensions=[ext], run=r)[0]
            tgts=np.loadtxt(opj('targets', accessions[subject]+'_run-%02d.txt'%r)).astype('int')                        
            ds = P.fmri_dataset(f.filename,
                             targets=tgts,
                             chunks=run)
            if not ds.shape[1]:
                raise ValueError("Got zero mask (no samples)")

            print "subject", subject, "chunk", run, "run", r, "ds", ds.shape 
            data.append(ds)
        data=P.vstack(data, a=0)
        if cache:
            P.Dataset.save(data, cache_filename, compression='gzip')
    return data

def inspect_bold_data(data):
    """
    Inspect Raw and Preprocessed Data (detrended / z-scored)
    Inspect original BOLD data

    inputs:
        data - the data to be inspected
    """
    ds=data.copy(deep=True)
    pl.figure()
    pl.plot(ds.samples[:,450:550])
    pl.title('Original BOLD data',fontsize=20)
    pl.xlabel('TR',fontsize=16)
    pl.ylabel('BOLD signal',fontsize=16)
    pl.axis('tight')
    pl.grid()

    # Inspect Detrended/Z-scored BOLD data
    ds2=ds.copy(deep=True)
    P.poly_detrend(ds2, polyord=1, chunks_attr='chunks') # in-place
    P.zscore(ds2, param_est=('targets', [1,2])) # in-place

    pl.figure()
    pl.plot(ds2.samples[:,450:550])
    pl.title('Detrended/z-scored BOLD data',fontsize=20)
    pl.xlabel('TR',fontsize=16)
    pl.ylabel('BOLD signal',fontsize=16)
    pl.axis('tight')
    _=pl.grid()

def do_subj_classification(ds, subject, task='timbre', condition='a', clf=None, null_model=False): #nf=50):
    """
    Classify a subject's data
    
    inputs:
            ds - a (masked) dataset
       subject - subject  - sid00[0-9]{4}    
          task - choose the clf task
     condition - choose the condition h/i/a
           clf - the classifier (LinearCSVMC)
    null_model - Monte-Carlo testing [False]

    outputs:
        dict = {
            'subj' : 'sid00[0-9]{4}'    # subject id
             'res' : [targets, predictions] # list of targets, predictions
              'cv' : CrossValidation results, including cv.sa.stats
            'task' : which task was performed
       'condition' : which condition in {'h','i'}
              'ut' : unique targets for task
      'null_model' : if results are for monte carlo testing [False]
       }    
    """
    clf=P.LinearCSVMC() if clf is None else clf
    P.poly_detrend(ds, polyord=1, chunks_attr='chunks') # in-place
    P.zscore(ds, param_est=('targets', [1,2])) # in-place
    ds = ds[(ds.targets>99) & (ds.targets<1000)] # take only pitch targets
    if task==tasks[1]: # pch-class
        ds.targets = (ds.targets % 100) % 12
    elif task==tasks[2]: # pch-hilo
        ds.targets = (ds.targets % 100)
        ds = ds[(ds.targets<=66) | (ds.targets>75)]
        ds.targets[ds.targets<=66]=1
        ds.targets[ds.targets>75]=2
    elif task==tasks[3]:  # timbre
        ds.targets = ds.chunks.copy() % 2
    if condition[0]=='h':
        ds = ds[(ds.chunks==1)|(ds.chunks==2)|(ds.chunks==5)|(ds.chunks==6)]
    elif condition[0]=='i':
        ds = ds[(ds.chunks==3)|(ds.chunks==4)|(ds.chunks==7)|(ds.chunks==8)]
    if null_model:
        ds.targets = np.random.permutation(ds.targets) # scramble pitch targets
    cv = P.CrossValidation(clf, P.HalfPartitioner(), errorfx=None, postproc=None) # Raw predictions
    res=cv(ds)
    return {'subj':subject, 'res':res, 'cv': cv, 'task':task, 'condition':condition, 'ut':ds.UT, 'null_model':null_model}

def mask_subject_ds(ds, subj, rois):
    """
    Mask a subject's data with a given list of rois
    
    inputs:
         ds - the dataset to mask
       subj - sid00[0-9]{4}
       rois - list of rois to merge e.g. [1005, 1035, 2005, 2035]
    
    outputs:
     ds_masked - the masked dataset (data is copied)
    """
    mask = get_subject_mask('sub-%s'%subj, run=1, rois=rois)
    mapper=P.mask_mapper(mask.samples.astype('i'))
    mapper.train(ds)
    ds_masked = ds.get_mapped(mapper)
    return ds_masked

def do_masked_subject_classification(ds, subj, task, cond, rois=[1030,2030], n_null=10, clf=None, show=False, null_model=False):
    """
    Apply mask and do_subj_classification

    inputs:
  
            ds - a (masked) dataset                                                                                  
       subject - subject  - sid00[0-9]{4}
          task - choose the clf task                                                                          
          rois - regions of interest to use
        n_null - how many Monte-Carlo runs to use if null_model
           clf - the classifier (LinearCSVMC)  
    null_model - Monte-Carlo testing [False]     
    """
    clf = P.LinearCSVMC() if clf is None else clf
    ds_masked = mask_subject_ds(ds, subj, rois)
    r=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=False)
    res=[r['res'].targets, r['res'].samples.flatten()]
    null=[]
    if null_model:
        for _ in range(n_null):
          n=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=True)
          null.append([n['res'].targets, n['res'].samples.flatten()])
    return res, null
          
def ttest_result(subj_res, task, roi, hemi, cond, n_null=10, clf=None, show=False, null_model=False):
    """
    Perform group statistical testing on subjects' predictions against baseline and null model conditions

    inputs:
     subj_res - the raw results (targets,predictions) for each subject, task, roi, hemi, and cond
         task - which task to generate group result for
          roi - which region of interest to use
         hemi - which hemisphere [0, 1000] -> L,R
         cond - heard: 'h' or imagined: 'i'
       n_null - number of Monte Carlo runs in null model [10]
          clf - classifier [LinearCSVMC]
         show - plot the results [False]
    null_model- whether to randomize targets [False]

    outputs:
      result dict - {
          'tt': t-test statistics
          'wx': wilcoxon test statistics
          'mn': per-subject within-subject mean, 
          'se': per-subject within-suject stanard error 
         'mn0': null model per-subject mean
         'se0': null model per-subject standard error
          'ut': unique targets
         }
    """
    res=[]
    null=[]
    clf = P.LinearCSVMC() if clf is None else clf
    hemiL = 'LH' if not hemi else 'RH'    
    for subj in subj_res.keys(): # subjects
        r=subj_res[subj][task][roi][hemiL][cond][0]
        res.append([r[0], r[1]])
        if null_model:
            for _ in range(n_null):
                n=subj_res[subj][task][roi][hemiL][cond][1]
                null.append([n[0], n[1]])
    res=np.array(res)
    a = (res[:,0,:]==res[:,1,:]).mean(1)
    ae = a.std() / np.sqrt(len(a))
    am=a.mean()
    if null_model:
        null=np.array(null)
        b = (null[:,0,:]==null[:,1,:]).mean(1)
        bm=b.mean()
        be = b.std() / np.sqrt(len(b))
    else:
        b=0.0
        bm=0.0
        be=0.0
    conds={'h':'Heard','i':'Imag'}
    L = 21
    ut = np.unique(r[0]) # targets
    bl = 1.0 / len(ut)
    tt = P.ttest_1samp(a, bl , alternative='greater') # pymvpa's ttest_1samp
    wx = wilcoxon(a-bl) # non-parametric version of ttest_1samp, **boosted by repeating a, N=22
    print "TT:(%4.1f, %0.6f)"%(tt[0],tt[1]),"WX:(%4.1f, %0.6f)"%(wx[0],wx[1])
    if show:
        pl.figure()
        if null_model:
            hst,bins,ptch=pl.hist(np.c_[a,b],np.linspace(0, 1, L+1), histtype='bar')
        else:
            hst,bins,ptch=pl.hist(a,np.linspace(0, 1, L+1), histtype='bar')            
        leg=['True','Null','Baseline'] if null_model else ['True','Baseline']
        pl.legend(leg,fontsize=16)
        pl.xlabel('Acc',fontsize=16)
        pl.ylabel('Freq',fontsize=16)
        pl.title('Task: %s(%s), T=%7.1f, p=%7.5f, mn=%5.3f/bl=%5.3f'%(task.upper(),conds[cond],tt[0],tt[1], a.mean(), bl),fontsize=20)
        ax=pl.gca()
        ax.set_xlim(0,1)
        mx = ax.get_ylim()[1]
        pl.plot([bl, bl],[0, 0.9*mx],'r--',alpha=0.5,linewidth=3)
        roi_str='ROI:'.join([' '+roi_map[roi].replace('ctx-','').upper()])
        pl.text(0.33,ax.get_ylim()[1]*0.925, roi_str, {'fontsize':20})
        pl.grid()
        pl.show()
    return {'tt':tt, 'wx':wx, 'mn':am, 'se':ae, 'mn0':bm, 'se0':be, 'ut': ut}

def calc_group_results(subj_res, group_res=None, null_model=False):
    """
    Calculate all-subject group results for tasks, rois, hemis, and conds
    Ttest and wilcoxon made relative to baseline of task

    inputs:
        subj_res - per-subject raw results (targets, predictions) per task,roi,hemi, and cond
        group_res- partial group-result dict to be updated / expanded [None]
      null_model - whether to use null model [False]

    outputs:
       group_res - group-level ttest / wilcoxon results over within-subject means
    """
    if group_res is None:
        group_res = {}
    subjects=subj_res.keys()
    for task in subj_res[subjects[-1]].keys():
        if task not in group_res.keys():
            group_res[task]={}
        for roi in subj_res[subjects[-1]][task].keys():
            if roi not in group_res[task].keys():
                group_res[task][roi]={}
                for hemi in [0,1000]:
                    hemiL = 'LH' if not hemi else 'RH'
                    group_res[task][roi][hemiL]={}
                    for cond in ['h','i']:
                        # show=True plots hist accuracy for conds x hemi with same bins
                        print task, roi_map[roi+hemi].replace('ctx-',''), cond.upper(),
                        group_res[task][roi][hemiL][cond] = ttest_result(subj_res, task, roi, hemi, cond, null_model=null_model)
    return group_res

def get_stars(mn,bl,p):
    """
    Utility function to return number of stars indicating level of significance:
          p<0.05: '*'
         p<0.005: '**'
        p<0.0005: '***'
    """
    stars=''
    if mn>bl: # one-sided significance
        if p<0.05: stars='*'
        if p<0.005: stars='**'
        if p<0.0005: stars='***'
    return stars

def plot_group_results(group_res, show_null=False, w=1.5):
    """
    GENERATE LOTS OF FIGURES, GROUP-LEVEL ANALYSIS for TASK x ROI x COND x HEMI          

    inputs:
       group_res - group results dict
    """
    dp = 3 if show_null else 2
    for task in group_res.keys():
        pl.figure(figsize=(24,6))
        pl.title(task,fontsize=20)
        xlabs = []
        pos=0
        mins=[]
        for roi in group_res[task].keys():
            for hemi in [0,1000]:
                hemiL = 'LH' if not hemi else 'RH'
                for cond in ['h','i']:
                    # Plot hist accuracy for conds x hemi with same bins
                    #task, roi_map[roi+hemi].replace('ctx-',''), cond.upper()
                    r=group_res[task][roi][hemiL][cond]
                    c=['blue','cyan'] if cond=='h' else ['purple','magenta']
                    pl.bar(pos, r['mn'], yerr=r['se'], color=c[0], width=w, align='center',alpha=0.75, ecolor='black', capsize=5)
                    mins.append(r['mn'])
                    if show_null:
                        pl.bar(pos+1, r['mn0'], yerr=r['se0'], color=c[1], width=w,align='center',alpha=0.75, ecolor='black', capsize=5)
                        mins.append(r['mn0'])
                    if cond=='h':
                        xlabs.append(roi_map[roi+hemi].replace('ctx-','').upper())
                    else:
                        xlabs.append('')
                    pos+=dp
        ax=pl.gca()
        mx=ax.get_ylim()[1]
        ax.set_ylim(np.array(mins).min()*0.95,mx*1.05)
        bl = 1.0 / len(r['ut'])
        pl.xticks(np.arange(len(xlabs))*dp+1.5,xlabs, rotation=90, fontsize=10)
        pl.ylabel('Mean Accuracy (N=%d)'%(len(subjects)), fontsize=18)
        pos=0
        leg = ['HD','HD0','IM','IM0'] if show_null else ['HD','IM']
        pl.legend(leg,fontsize=18,loc=2)
        pl.plot([-dp,len(xlabs)*dp+1],[bl,bl],'g--',linewidth=3)
        pl.text(len(xlabs)*dp+1.,bl*1.01,'baseline',fontsize=16)
        pl.grid(linewidth=0.5)
        rnge = ax.get_ylim()
        rnge = rnge[1]-rnge[0] # get vertical range for scaling
        for roi in group_res[task].keys():
            for hemi in [0,1000]:
                hemiL = 'LH' if not hemi else 'RH'
                for cond in ['h','i']:
                    r=group_res[task][roi][hemiL][cond]
                    if not np.isnan(r['tt'][0]) and r['tt'][0]>0:
                        p = r['tt'][1] # ttest 1samp
                        stars = get_stars(r['mn'],bl, p)
                        pl.text(pos-0.5-0.666*len(stars), r['mn']+rnge*0.1, stars, color='k', fontsize=20)
                    if not np.isnan(r['wx'][0]) and r['wx'][0]>0:
                        p = r['wx'][1] #  wilcoxon 1samp
                        stars = get_stars(r['mn'],bl, p)
                        pl.text(pos-0.5-0.666*len(stars), r['mn']+rnge*0.125, stars, color='r', fontsize=20)
                    pos+=dp


def save_result_subj_task(res, subject, task):
    # Save partial (subj,task) results data
    with open(opj(OUTDIR, "%s_%s_res_part.pickle"%(subject,task)), "wb") as f:
        pickle.dump(res, f)

def load_all_subj_res_from_parts():
    subj_res={}
    for subj in subjects:
        subj_res[subj]={}
        for task in tasks:
            with open(opj(OUTDIR, "%s_%s_res_part.pickle"%(subj,task)), "rb") as f:            
                res_part = pickle.load(f)
                subj_res[subj].update(res_part[subj])
    return subj_res

if __name__=="__main__":
    if len(sys.argv) < 3:
        print "usage: %s sid00[0-9]{4} task{pch-height|pch-class|pch-hilo|timbre}"%sys.argv[0]
        sys.exit(1)
    subj = sys.argv[1]
    task = sys.argv[2]

    if subj not in subjects:
        print "%s not in subjects"%subj
        print "subjects:", subjects
        sys.exit(1)        

    if task not in tasks:
        print "%s not in tasks"%task
        print "tasks:", tasks
        sys.exit(1)        
        
    # Cortical regions of interest, group_results are L-R lateralized with R=roi_id + 1000
    rois = range(1000,1036) #[1001, 1012, 1014, 1015, 1022, 1023, 1024, 1027, 1028, 1030, 1031, 1034, 1035]
    rois.pop(rois.index(1004)) # corpuscallosum is degenerate, so remove it from the list of ROIs
    ds=get_subject_ds(subj)
    res={}
    res[subj]={}
    res[subj][task]={}
    for roi in rois:
        res[subj][task][roi]={}
        for hemi in [0,1000]:                          
            hemiL = 'LH' if not hemi else 'RH'
            res[subj][task][roi][hemiL]={}            
            for cond in ['h','i']:
                res[subj][task][roi][hemiL][cond]=do_masked_subject_classification(ds, subj, task, cond, [roi+hemi])
    # Save partial (subj,task) results to intermediate result file
    save_result_subj_task(res, subj, task)
    
