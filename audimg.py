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
from scipy.stats import wilcoxon, ttest_rel
import sys
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.linear_model import Lasso
import pdb
import pprint

pl = P.pl # convenience for access to plotting functions

# data = 3mmx3mmx3mm

ROOTDIR='/isi/music/auditoryimagery2'
DATADIR=opj(ROOTDIR, 'am2/data/fmriprep/fmriprep/')
OUTDIR=opj(ROOTDIR, 'results_audimg_subj_task')

MRISPACE= 'MNI152NLin2009cAsym' # if using fmriprep_2mm then MRISPACE='MNI152NLin6Asym' 
PARCELLATION='desc-aparcaseg_dseg'# per-subject ROI parcellation in MRISPACE

# List of tasks to evaluate
tasks=['pch-height','pch-class','pch-hilo','timbre','pch-helix-stim-enc']

def _make_subj_id_maps():
    """
    Utility function
    Read subj-id-accession-key.csv to map ids to accession number and tonality (E or F)
    """
    global subjects, accessions, tonalities, subjnums
    subjects = []
    accessions = {}
    tonalities = {}
    subjnums = {}
    with open(opj(ROOTDIR,'subj-id-accession-key.csv')) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row = reader.next() # header row
        for row in reader:
            subj_num, subj_id, accession_num, key = row
            subj_id = subj_id.lower()
            #print subj_num, subj_id, accession_num, key
            subjects.append(subj_id)
            subjnums[subj_num] = subj_id
            accessions[subj_id] = accession_num
            tonalities[subj_id] = key 

_make_subj_id_maps()

def _print_subj_id_maps():
    """
    Utility functino
    pretty print subject id / num / accession / key maps
    """    
    pprint.pprint(subjnums)            
    pprint.pprint(subjects)    
    pprint.pprint(accessions)
    pprint.pprint(legend)    
    pprint.pprint(tonalities)
    pprint.pprint(tasks)
    
def _make_legend(LEGEND_FILE=opj(ROOTDIR,"subj_task_run_list.txt")):
    """
    Utility function
    Read run_legend to re-order runs by task
    We could hard-code this as a dict, only need the first row for run-order
    """
    global legend
    legend = {}
    with open(LEGEND_FILE,"rt") as f:
        try:
            lines = [line.strip() for line in f.readlines()]
        except:
            raise IOError("Cannot open %s for reading"%LEGEND_FILE)
        for line in lines:
            line = line.split('-') # sid001125_task - pitchheardXtrumXF_run - 01
            subj_id = line[0].split('_')[0]
            run = int(line[-1])
            task=line[1][5].upper() # H or I     
            task+=line[1].split('X')[1][0].upper() # T or C     
            if run==1:
                legend[accessions[subj_id]]=[]
            legend[accessions[subj_id]].append(task)

def _make_legend_deprecated():
    """
    Utility function
    Read run_legend to re-order runs by task
    We could hard-code this as a dict, only need the first row for run-order
    """
    global legend
    with open(opj(ROOTDIR,'targets/run_legend.csv')) as csvfile:
        legend={}
        for subj, acc in accessions.items():
            legend[acc]=[]
        reader = csv.reader(csvfile, delimiter=',')    
        for row in reader:
            legend[row[0]].append(row[2])

_make_legend()

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

def _gen_RH_cortical_map():
    """
    Utility function to generate RH of cortical map
    """
    roi_map.pop(1004) # The Corpus Callosum is not defined
    for k in roi_map.keys():
        roi_map[k+1000]=roi_map[k].replace('lh','rh')

_gen_RH_cortical_map()

def get_LH_roi_keys():
    """
    Return the LH ROI keys, methods will auto-lateralize
    """
    return sorted (roi_map.keys())[:len(roi_map.keys())/2] # LH only
    
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
            #print "subject", subject, "chunk", run, "run", r, "ds", ds.shape 
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

def _encode_targets(ds,subj,task):
    """
    Utility function
    Given a dataset ds, subj, and task, return ds with target encoding for task
    subj is required to map tonalities ('E' or 'F') onto relative pc index
    """
    ds = ds[(ds.targets>99) & (ds.targets<1000)] # take only pitch targets    
    if task==tasks[1]: # pch-class
        key_ref = 52 if tonalities[subj]=='E' else 53
        ds.targets -= key_ref # shift to relative common reference
        ds.targets = (ds.targets % 100) % 12
    elif task==tasks[2]: # pch-hilo
        ds.targets = (ds.targets % 100)
        ds = ds[(ds.targets<=66) | (ds.targets>75)]
        ds.targets[ds.targets<=66]=1
        ds.targets[ds.targets>75]=2
    elif task==tasks[3]:  # timbre
        ds.targets = ds.chunks.copy() % 2
    return ds

def _map_pc_to_helix(ds_regr, subj):
    """
    Utility function
    Map targets into pitch-class helix
    inputs:
      ds_regr - regression masked dataset (reversed)
        subj  - subject id sid00[0-9]{4}
    outputs:
       ds_cv  - dataset with helix-mapped pitch-class predictors
    """
    fifths_map={0:0,  2:2,  4:4,  5:-1,  7:1,  9:3, 11:5} # encode pc as 5ths
    pc_idx = np.where((ds_regr.samples>100) & (ds_regr.samples<500))[0]
    ds_cv = ds_regr[pc_idx]
    # Subject tonality / key
    key_ref = 52 if tonalities[subj]=='E' else 53
    ds_cv.samples -= key_ref
    ds_cv.samples %= 100
    pcs = ds_cv.samples.flatten()
    f = np.array([fifths_map[k%12] for k in pcs])
    c = np.cos(f*np.pi/2)
    s = np.sin(f*np.pi/2)
    h = pcs / 12.
    h -= h.mean()
    h /= h.std()
    ds_cv.samples = np.c_[c,s]#,h]
    return ds_cv, pc_idx

def _get_stimulus_encoding_ds(ds, subj):
    """
    Utility function
    Make target-encoded dataset, stimulus-encoding dataset for subject
    """
    column = np.argmax(np.abs(ds.samples).mean(0)) # per-voxel modeling, so choose a column (voxel) UNIVARIATE
    ds_regr = P.Dataset(ds.targets, sa={'targets':ds.samples[:,column],'chunks':ds.chunks})
    ds_cv, pc_idx = _map_pc_to_helix(ds_regr, subj) # ds_cv is targets -> 1 voxel
    ds = ds[pc_idx] # truncate masked dataset, with all voxels, to pc_idx
    return ds, ds_cv

def do_stimulus_encoding(ds, subj, clf=SKLLearnerAdapter(Lasso(alpha=0.2))):
    """
    Regression to predict a subject's BOLD response to stimulus
    
    inputs:
            ds - a (masked) dataset
          subj - subject id sid00[0-9]{4}
           clf - regression model [SKLLearnerAdapter(Lasso(alpha=0.1))]

    outputs:    
           ds.targets - RMSE values for model
           ds.samples - RMSE values for null
    """
    ds, ds_cv = _get_stimulus_encoding_ds(ds, subj)    
    cv = P.CrossValidation(clf, P.HalfPartitioner(), postproc=None, errorfx=P.rms_error)
    # Compare regression model and permutation null
    res=[]
    null=[]
    for voxel in ds.samples.T:
        ds_cv.targets = voxel
        r=cv(ds_cv)
        res.append(r.samples.mean())
        ds_cv.targets = np.random.permutation(voxel) # randomly permute targets
        n=cv(ds_cv)
        null.append(n.samples.mean())
    res_ds = P.Dataset(np.array(null), sa={'targets': np.array(res) })
    return res_ds
                                                                                                        
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
    ds = _encode_targets(ds, subject, task)
    clf=P.LinearCSVMC() if clf is None else clf        
    cv = P.CrossValidation(clf, P.HalfPartitioner(), errorfx=None, postproc=None) # Raw predictions
    
    if condition[0]=='h':
        ds = ds[(ds.chunks==1)|(ds.chunks==2)|(ds.chunks==5)|(ds.chunks==6)]
    elif condition[0]=='i':
        ds = ds[(ds.chunks==3)|(ds.chunks==4)|(ds.chunks==7)|(ds.chunks==8)]
    if null_model:
        ds.targets = np.random.permutation(ds.targets) # scramble pitch targets
    if task=='pch-helix-stim-enc':
        res=do_stimulus_encoding(ds, subject)
    else:
        res=cv(ds)
    return {'subj':subject, 'res':res, 'cv': cv, 'task':task, 'condition':condition, 'ut':ds.UT, 'null_model':null_model}

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
    fname = opj(path, 'sub-%s'%subject, 'func', 'sub-%s_task-*_run-%02d_space-%s_%s.nii.gz'%(subject,run,space, parcellation))
    #print fname
    fname = glob.glob(fname)[0]
    ds=P.fmri_dataset(fname)
    found = np.where(np.isin(ds.samples,rois))[1]
    return ds[:,found]

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
    mask = get_subject_mask('%s'%subj, run=1, rois=rois)
    ds_masked=P.fmri_dataset(P.map2nifti(ds), ds.targets, ds.chunks, P.map2nifti(mask))
    P.poly_detrend(ds_masked, polyord=1, chunks_attr='chunks') # in-place
    P.zscore(ds_masked, param_est=('targets', [1,2])) # in-place    
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

    outputs:
          [targets, predictions], [[null_targets1,null_predictions1], ...]
    """
    clf = P.LinearCSVMC() if clf is None else clf
    ds_masked = mask_subject_ds(ds, subj, rois)
    r=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=False)
    res=[r['res'].targets, r['res'].samples.flatten()]
    null=[]
    if null_model: # for classifiers only
        for _ in range(n_null):
          n=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=True)
          null.append([n['res'].targets, n['res'].samples.flatten()])
    if task == 'pch-helix-stim-enc':        
        null = [res[1],[]] # we need the null for stats testing, no baseline
        res = [res[0],[]]
    return res, null
          
def ttest_result(subj_res, task, roi, hemi, cond, n_null=10, null_model=False):
    """
    Perform group statistical testing on subjects' predictions against baseline and null model conditions

    inputs:
     subj_res - the raw results (targets,predictions) for each subject, task, roi, hemi, and cond
         task - which task to generate group result for
          roi - which region of interest to use
         hemi - which hemisphere [0, 1000] -> L,R
         cond - heard: 'h' or imagined: 'i'
       n_null - number of Monte Carlo runs in null model [10]
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
    if task != 'pch-helix-stim-enc':
        res=[]
        null=[]
        hemiL = 'LH' if not hemi else 'RH'    
        for subj in subj_res.keys(): # subjects
            r=subj_res[subj][task][roi][hemiL][cond][0]
            res.append([r[0], r[1]])
            if null_model:
                for _ in range(n_null):
                    n=subj_res[subj][task][roi][hemiL][cond][1]
                    null.append([n[0], n[1]])
        res=np.array(res)
        a = (res[:,0,:]==res[:,1,:]).reshape(2,-1).mean(1) # Half Partitioner result
        ae = a.std() / np.sqrt(len(a))
        am=a.mean()
        if null_model:
            null=np.array(null)
            b = (null[:,0,:]==null[:,1,:]).reshape(2,-1).mean(1) # Half Partitioner result
            bm=b.mean()
            be = b.std() / np.sqrt(len(b))
        else:
            b=0.0
            bm=0.0
            be=0.0
        ut = np.unique(r[0]) # targets
        bl = 1.0 / len(ut)
        tt = P.ttest_1samp(a, bl , alternative='greater') # pymvpa's ttest_1samp
        wx = wilcoxon(a-bl) # non-parametric version of ttest_1samp, **boosted by repeating a, N=22
        #print "TT:(%4.1f, %0.6f)"%(tt[0],tt[1]),"WX:(%4.1f, %0.6f)"%(wx[0],wx[1])
    else:
        res=[]
        null=[]
        hemiL = 'LH' if not hemi else 'RH'    
        for subj in subj_res.keys(): # subjects
            r=subj_res[subj][task][roi][hemiL][cond]
            res.append([r[0][0].mean(), r[1][0].mean()])
        res=np.array(res)
        tt = ttest_rel(res[:,0],res[:,1]) # model vs null
        wx = wilcoxon(res[:,0],res[:,1]) # model vs null
        am = res[:,0].mean()
        ae = res[:,0].std() / np.sqrt(len(res))
        bm = res[:,1].mean()
        be = res[:,1].std() / np.sqrt(len(res))
        ut = np.array([0,2,4,-1,1,3,5]) # pcs as 5ths
    return {'tt':tt, 'wx':wx, 'mn':am, 'se':ae, 'mn0':bm, 'se0':be, 'ut': ut}

def ttest_per_subj_res(subj_res):
    """
    for each subject, perform a 1-sample t-test on the per-target, per-run accuracies against baseline
    return t-test result dict
    """
    rois = get_LH_roi_keys()
    res={}
    for subj in subjects:
        res[subj]={}
        for task in tasks[:-1]:
            res[subj][task]={}            
            for roi in rois:
                res[subj][task][roi]={}
                for hemi in ['LH','RH']:
                    res[subj][task][roi][hemi]={}                
                    for cond in ['h','i']:                    
                        r=subj_res[subj][task][roi][hemi][cond][0]      
                        ut = np.unique(r[0])
                        # targets
                        # How many targets per run
                        test = []
                        tgts = []
                        # split into HalfPartitioner sets and evaluate per set
                        for run in zip(np.array(r[0]).reshape(2,-1), np.array(r[1]).reshape(2,-1)):
                            for tgt in np.unique(run[0]): # use only targets for this run
                                idx = np.where(run[0]==tgt)[0] # targets within run
                                test.append((run[1][idx]==run[0][idx]).mean())  # precision: predictions mean-acc this target/run
                                tgts.append(1./len(idx)) # 1/Nt 
                        test = np.array(test)
                        tgts = np.array(tgts)
                        x = test
                        y = tgts
                        if np.any(np.isinf(x) | np.isinf(y) | np.isnan(x) | np.isnan(y)):
                            pdb.set_trace()
                        tt=P.ttest_1samp(x, y.mean()) # accuracy vs baseline, per target, per run
                        wx=wilcoxon(x-y.mean())
                        res[subj][task][roi][hemi][cond]={'TP':test, 'P':tgts, 'mn':x.mean(), 'se':x.std()/np.sqrt(len(x)),
                                                    'm0':y.mean(), 'se0':y.std()/np.sqrt(len(y)),
                                                          'ut':ut, 'tt':tt,'wx':wx, 'baseline': y.mean()}
    return res

def count_sub_sig_res(subj_res):
    """
    Count group significant results
    inputs:
        subj_res - per-(subject, task, roi, hemi, cond) classifier significance tests
    outputs:
        group_res - per-(task, roi, hemi, cond) significant result, punning the 'mn' field as counts
    """
    rois = get_LH_roi_keys()
    sig_res = {}
    for task in subj_res[subjects[0]].keys(): # may be short a key
        sig_res[task]={}
        for roi in rois:
            sig_res[task][roi]={}
            for hemi in ['LH','RH']:
                sig_res[task][roi][hemi]={}
                for cond in ['h','i']:
                    s = sig_res[task][roi][hemi][cond] = {'mn':0, 'se':0,'m0':0,
                                                          'se0':0,'ut':[0],
                                                          'tt':np.array([0.,0.]),'wx':np.array([0.,0.])}
                    for subj in subjects:
                        r = subj_res[subj][task][roi][hemi][cond]
                        if r['tt'][0]>0 and (r['tt'][1]<0.05 or r['wx'][1]<0.05) :
                            s['mn']+=1 # Punning the mean field as count
                            s['se']+=r['se']
                            s['m0']+=r['m0']
                            s['se0']+=r['se0']
                            s['ut']+=r['ut']
                            s['tt']+=np.array([r['tt'][0],r['tt'][1]])
                            s['wx']+=np.array([r['wx'][0],r['wx'][1]])                            
                    if s['mn']: # normalize for mean values over significant results
                        s['se']/=s['mn']
                        s['m0']/=s['mn']
                        s['se0']/=s['mn']
                        s['ut']/=s['mn']
                        s['tt']/=s['mn']
                        s['wx']/=s['mn']
    return sig_res


# def count_sub_sig_res(subj_res):
#     """
#     Alternative group summary, counting of significant results
#     """
#     rois = get_LH_roi_keys()
#     sig_res = {}
#     for task in tasks[:-1]:
#         sig_res[task]={}
#         for roi in rois:
#             sig_res[task][roi]={}
#             for hemi in ['LH','RH']:
#                 sig_res[task][roi][hemi]={}
#                 for cond in ['h','i']:
#                     sig_res[task][roi][hemi][cond]={'mn':0, 'se':0,
#                                                     'm0':0, 'se0':0,'ut':[0],
#                                                     'tt':np.array([0.,0.]),'wx':np.array([0.,0.])}
#                     for subj in subjects:
#                         a,b=subj_res[subj][task][roi][hemi][cond][0]
#                         tt = ttest_rel(a,b) # NOT CORRECT, WHAT IS THIS RESULT?
#                         if tt[0]>0 and tt[1]<0.05:
#                             sig_res[task][roi][hemi][cond]['mn']+=1
#                             sig_res[task][roi][hemi][cond]['tt']+=np.array([tt[0],tt[1]])
#                             wx = wilcoxon(a,b)
#                             sig_res[task][roi][hemi][cond]['wx']+=np.array([wx[0],wx[1]]) 
#                     if sig_res[task][roi][hemi][cond]['mn']:
#                         sig_res[task][roi][hemi][cond]['tt']/=sig_res[task][roi][hemi][cond]['mn']
#                         sig_res[task][roi][hemi][cond]['wx']/=sig_res[task][roi][hemi][cond]['mn']               
#     return sig_res

def calc_group_results(subj_res, null_model=False):
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
    group_res = {}
    subjects=subj_res.keys()
    for task in subj_res[subjects[0]].keys():
        group_res[task]={}
        for roi in subj_res[subjects[0]][task].keys():
            group_res[task][roi]={}
            for hemi in [0,1000]:
                hemiL = 'LH' if not hemi else 'RH'
                group_res[task][roi][hemiL]={}
                for cond in ['h','i']:
                    #print task, roi_map[roi+hemi].replace('ctx-',''), cond.upper(),
                    group_res[task][roi][hemiL][cond] = ttest_result(subj_res, task, roi, hemi, cond, null_model=null_model)
    return group_res

def _get_stars(mn,bl,p, stim_enc=False):
    """
    Utility function to return number of stars indicating level of significance:
          p<0.05: '*'
         p<0.005: '**'
        p<0.0005: '***'
    """
    stars=''
    if not stim_enc:
        if mn>bl: # one-sided significance
            if p<0.05: stars='*'
            if p<0.005: stars='**'
            if p<0.0005: stars='***'
    else:
        if mn<bl: # one-sided significance
            if p<0.05: stars='*'
            if p<0.005: stars='**'
            if p<0.0005: stars='***'
    return stars

def plot_group_results(group_res, show_null=False, w=1.5, is_counts=False, ttl=''):
    """
    Generate figures for group-level analysis: TASK,  ROI x COND x HEMI          

    inputs:
       group_res - group results dict
    """
    dp = 3 if show_null else 2
    for task in sorted(group_res.keys()):
        pl.figure(figsize=(24,6))
        pl.title(ttl+' '+task,fontsize=20)
        xlabs = []
        pos=0
        mins=[]
        for roi in get_LH_roi_keys():
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
        if task == 'pch-helix-stim-enc':
            if 'baseline' in r:
                bl = r['baseline']
            else:
                bl = r['mn0']
        else:
            bl = 1.0 / len(r['ut'])
        ax.set_ylim(min(bl,np.array(mins).min())*0.95,mx*1.05)
        pl.xticks((np.arange(len(xlabs))+0.5)*dp,xlabs, rotation=90, fontsize=16)
        if task=='pch-helix-stim-enc':
            pl.ylabel('Mean Root-Mean-Square Err (N=%d)'%(len(subjects)), fontsize=18)
        elif is_counts:
            pl.ylabel('# subjects (N=%d)'%(len(subjects)), fontsize=18)            
        else:
            pl.ylabel('Mean Accuracy (N=%d)'%(len(subjects)), fontsize=18)
        pos=0
        leg = ['HD','HD0','IM','IM0'] if show_null else ['HD','IM']
        pl.legend(leg,fontsize=18,loc=2)
        if not is_counts:
            pl.plot([-dp,len(xlabs)*dp+1],[bl,bl],'g--',linewidth=3)
            pl.text(len(xlabs)*dp+1.,bl*1.01,'baseline',fontsize=16)
        pl.grid(linewidth=0.5)
        rnge = ax.get_ylim()
        rnge = rnge[1]-rnge[0] # get vertical range for scaling
        for roi in get_LH_roi_keys():
            for hemi in [0,1000]:
                hemiL = 'LH' if not hemi else 'RH'
                for cond in ['h','i']:
                    r=group_res[task][roi][hemiL][cond]
                    if not np.isnan(r['tt'][0]) and r['tt'][0]>0:
                        p = r['tt'][1] # ttest 1samp
                        stars = _get_stars(r['mn'],bl, p, task=='pch-helix-stim-enc')
                        pl.text(pos-0.666*len(stars), (r['mn']+r['se'])+rnge*0.05, stars, color='k', fontsize=12)
                    if not np.isnan(r['wx'][0]) and r['tt'][0]>0: # wx is not signed, so use tt for effect sign
                        p = r['wx'][1] #  wilcoxon 1samp
                        stars = _get_stars(r['mn'],bl, p, task=='pch-helix-stim-enc')
                        pl.text(pos-0.666*len(stars), (r['mn']+r['se'])+rnge*0.075, stars, color='r', fontsize=12)
                    pos+=dp


def save_result_subj_task(res, subject, task):
    """
    Save partial (subj,task) results data from a classifier
    
    inputs:
        res  - results output of do_subj_classification 
     subject - subject id - sid00[0-9]{4}
      task   - name of task from tasks
    
    outputs:
        saves file in OUTDIR "%s_%s_res_part.pickle"%(subject,task)
    """
    with open(opj(OUTDIR, "%s_%s_res_part.pickle"%(subject,task)), "wb") as f:
        pickle.dump(res, f)

def load_all_subj_res_from_parts():
    """
    Load all partial result files and concatenate into a single dict
    
    inputs: None, expects files in OUTDIR

    outputs:
       subj_res - per-subject results dict, indexed by sid00[0-9]{4}
    """
    subj_res={}
    for subj in subjects:
        subj_res[subj]={}
        for task in tasks:
            with open(opj(OUTDIR, "%s_%s_res_part.pickle"%(subj,task)), "rb") as f:            
                res_part = pickle.load(f)
                subj_res[subj].update(res_part[subj])
    return subj_res

if __name__=="__main__":
    """
    Classify subject BOLD data using task for all ROIs and save subject's results

    Usage: python audimg sid00[0-9]{4} task{pch-height|pch-class|pch-hilo|timbre}    
    """
    if len(sys.argv) < 3:
        print "Usage: %s sid00[0-9]{4} task{pch-height|pch-class|pch-hilo|timbre}"%sys.argv[0]
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

    TESTING=False
    if TESTING:
        print("Test OK: skipping save...")
        _print_subj_id_maps()
    else:
        # Cortical regions of interest, group_results are L-R lateralized with R=roi_id + 1000
        rois = get_LH_roi_keys()
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

    
