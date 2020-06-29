"""
audimg.py
Decoding Model for Heard / Imagined Pitch and Timbre
Michael A. Casey, Dartmouth College, Aug-Dec 2019

Library for auditory imagery fMRI experiment
Data preprocessing, classifier training, result grouping, result plotting
"""

import mvpa2.suite as P
import numpy as np
import bids.grabbids as gb
import os
import os.path
from os.path import join as opj
import csv
import glob
import pickle
from scipy.stats import wilcoxon, ttest_rel
import sys
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
import pprint
#import pdb

pl = P.pl # convenience for access to plotting functions

# data = 3mmx3mmx3mm

ROOTDIR='/isi/music/auditoryimagery2'
DATADIR=opj(ROOTDIR, 'am2/data/fmriprep/fmriprep/')
OUTDIR=opj(ROOTDIR,'results_audimg_subj_task_mkc_del1_dur3_n100')

MRISPACE= 'MNI152NLin2009cAsym' # if using fmriprep_2mm then MRISPACE='MNI152NLin6Asym' 
PARCELLATION='desc-aparcaseg_dseg'# per-subject ROI parcellation in MRISPACE

N_NULL=10 # number of null models to run

# List of tasks to evaluate
tasks=['pch-height','pch-class','pch-hilo','timbre','pch-helix-stim-enc','pch-classX','timbreX']

def set_outdir(outdir, rootdir=ROOTDIR, update=True):
    global OUTDIR
    if update:
        OUTDIR=opj(ROOTDIR, outdir)
    else:
        return opj(ROOTDIR, outdir)

def _make_subj_id_maps():
    """
    Utility function
    Read subj-id-accession-key.csv to map ids to accession number and tonality (E or F)

    Note: The following subjects have been removed from the control file: subj-id-accession-key.csv
          on the basis of behavioural measures (probe tone ratings, vividness scale)
    Red flagged:
    SID001415 	A002652
    SID000388 	A002655
    SID001564 	A003037
    SID001594 	A003098
    SID001613 	A003232
    SID001679 	A003272

    Note: The following may also need to be removed on the basis of behavioural measures (probe tone ratings, vividness scale)
    Yellow flagged:
    SID001125 	A003243
    SID001660 	A003231
    """
    global subjects, accessions, tonalities, subjnums
    subjects = []
    accessions = {}
    tonalities = {}
    subjnums = {}
    with open(opj(ROOTDIR,'subj-id-accession-key.csv')) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row = reader.next() # header row
        for row in reader: # remaining rows
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
1018:    "ctx-lh-parsopercularis",        #dcb48c # BA44
1019:    "ctx-lh-parsorbitalis",  #146432         # BA47
1020:    "ctx-lh-parstriangularis",       #dc3c14 # BA45
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


def get_subject_ds(subject, cache=False, cache_dir='ds_cache'):
    """Assemble pre-processed datasets    
    load subject original data (no mask applied)
    optionally cache for faster loading during model training/testing
    purpose: re-use unmasked dataset, applying mask when necessary    

    inputs:
        subject  - sid00[0-9]{4}    
        cache    - whether to use cached datasets [False]
     cache_dir   - where to store / load cached datasets ['ds_cache']

    outputs:
        data     - subject original data (no mask applied)
    """
    swap_timbres=[2,1,4,3,6,5,8,7]
    layout = gb.BIDSLayout(DATADIR)
    ext = 'desc-preproc_bold.nii.gz'
    cache_filename = '%s/%s.ds_cache.nii.gz'%(cache_dir, subject)
    cache_lockname = '%s/%s.ds_cache.lock'%(cache_dir, subject)    
    cache_fail=False
    if cache:
        try:
            data=P.Dataset.from_hdf5(cache_filename)        
        except:
            cache_fail=True
    if not cache or cache_fail:
        data=[]
        for run in range(1,9):
            r=run if legend[accessions[subject]][0]=='HT' else swap_timbres[run-1]
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
        if cache and not os.path.exists(cache_lockname):
            with open(cache_lockname, "w") as f:
                f.write("audimg lock\n");
                f.close()
                P.Dataset.save(data, cache_filename, compression='gzip')
                os.remove(cache_lockname)
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

def _encode_task_condition_targets(ds, subj, task, cond, delay=0, dur=1):
    """
    Utility function
    Given a dataset ds, subj, and task, return ds with target encoding for task
    subj is required to map tonalities ('E' or 'F') onto relative pc index
    ds    - masked dataset
    subj  - subject key
    task  - one of tasks[....]
    cond  - heard or imagined 'h' or 'i' 
    delay - TRs delay for target conditions          [0]
    dur   - event duration for event-related dataset [1]
    """
    ds = ds.copy()

    if delay>0: # shift the targets relative to the BOLD response
        ds.targets = np.r_[ np.zeros(delay), ds.targets[:-delay] ]

    idx = np.where( (ds.targets>99) & (ds.targets<1000) )[0]

    if dur==1:
        ds = ds[idx] # take only pitch targets 
    else: 
        onsets = idx.reshape(-1,2)[:,0] 
        events = [{'onset':on, 'duration':dur} for on in onsets]
        ds = P.extract_boxcar_event_samples(ds, events) # make event-related dataset P.eventrelated_dataset(ds, events) 
        ds.targets = np.array([t[0] for t in ds.targets])
        ds.chunks = np.array([c[0] for c in ds.chunks])
        
    if 'pch-class' in task: 
        key_ref = 52 if tonalities[subj]=='E' else 53
        ds.targets -= key_ref # shift to relative common reference
        ds.targets = (ds.targets % 100) % 12
    elif 'pch-hilo' in task: 
        ds.targets = (ds.targets % 100)
        ds = ds[(ds.targets<=66) | (ds.targets>75)]
        ds.targets[ds.targets<=66]=1
        ds.targets[ds.targets>75]=2
    elif 'timbre' in task:
        ds.targets = ds.chunks.copy() % 2
    if 'X' not in task: # preserve 'h' and 'i' if cross-decoding
        if cond[0]=='h':
            ds = ds[np.isin(ds.chunks, [1,2,5,6])]            
        elif cond[0]=='i':
            ds = ds[np.isin(ds.chunks, [3,4,7,8])]                    
    return ds

def _map_pc_to_helix(ds_regr, subj, height=False):
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
    # Subject tonality / key
    if subj is not None:
        key_ref = 52 if tonalities[subj]=='E' else 53
        ds_regr.samples -= key_ref
    else:
        ds_regr.samples -= 100 # test data
    ds_regr.samples %= 100
    pcs = ds_regr.samples.flatten()
    f = np.array([fifths_map[k%12] for k in pcs])
    c = np.cos(f*np.pi/2)
    s = np.sin(f*np.pi/2)
    if height:
        h = pcs / 12.
        h -= h.mean()
        h /= h.std()
        ds_regr.samples = np.c_[c,s,h]
    else: # just a circle
        ds_regr.samples = np.c_[c,s]
    return ds_regr

def _get_stimulus_encoding_ds(ds, subj):
    """
    Utility function
    Make target-encoded dataset, stimulus-encoding dataset for subject
    """
    column = np.argmax(np.abs(ds.samples).mean(0)) # per-voxel modeling, so choose a column (voxel) UNIVARIATE
    ds_regr = P.Dataset(ds.targets, sa={'targets':ds.samples[:,column],'chunks':ds.chunks})
    ds_regr = _map_pc_to_helix(ds_regr, subj) # ds_cv is targets -> 1 voxel
    return ds_regr

def test_stimulus_encoding():
    """
    Function to test stimulus encoding model
    """
    t=[]
    subj = None # None == test subject
    for _ in range(2):
        t.append(np.random.permutation( np.array([0,2,4,5,7,9,11]).repeat(12) )[:, np.newaxis])
    ds_circ=_map_pc_to_helix(P.dataset_wizard(np.vstack(t)), subj)
    t = np.vstack(t) + 100 # targets encoding
    w = np.random.randn(2,10) # latent cicular encoding weights
    s = np.dot(ds_circ.samples, w)
    ds_t = P.dataset_wizard(s, t.flatten(), np.r_[np.ones(84), np.ones(84)+1])
    res = do_masked_subject_classification(ds_t, subj, task='pch-helix-stim-enc', cond='h', rois=[1030])
    print "recon err:", P.errorfx.relative_rms_error(res[0][1],res[0][0])
    print "null err:", np.array([P.errorfx.relative_rms_error(res[1][n][1],res[0][0]) for n in range(10)]).mean()
    return res

def do_stimulus_encoding(ds, subj, clf=SKLLearnerAdapter(Lasso(alpha=0.2)), part=0, null_model=False):
    """
    Utility function: CV half partitioner stimulus-encoding model
    Regression to predict a subject's BOLD response to stimulus representation    

    inputs:
            ds - a (masked) dataset
          subj - subject id sid00[0-9]{4}
           clf - regression model [SKLLearnerAdapter(Lasso(alpha=0.1))]
          part - partition (0 or 1)
    null_model - whether using monte carlo tests [False]

    outputs:    
          tgts - BOLD data
          pred - predicted BOLD data
    """
    ds_cv = _get_stimulus_encoding_ds(ds, subj)    # target-filtered ds, swapped ds
    n=len(ds)
    ds.partitions = (np.arange(n)>=n/2).astype(int)
    if null_model:
        # Separate permuted training targets from true targets used for testing
        ds.samples[ds.partitions==part] = np.random.permutation(ds.samples[ds.partitions==part]) 
    tgts=[]
    preds=[]
    for voxel in ds.samples.T:
        ds_cv.targets = voxel
        clf.train(ds_cv[ds.partitions==part])    
        pred = clf.predict(ds_cv[ds.partitions!=part])
        tgts.append(voxel[ds.partitions!=part])
        preds.append(pred)
    return np.array(tgts).T, np.array(preds).T

def _cv_run(ds, clf, part=0, null_model=False):            
    """
    Utility function: CV half partitioner with probability estimates
    (resolves incompatibility between P.SVM and P.CrossValidation)

    inputs:
            ds - a (masked) dataset
           clf - regression model [SKLLearnerAdapter(Lasso(alpha=0.1))]
          part - partition (0 or 1)
    null_model - whether using monte carlo tests [False]

    outputs:    
          tgts - BOLD data
          pred - predicted BOLD data    
           est - probabilities of predicted labels
    """
    # 'null_model' : whether using monte carlo tests [False]
    n=len(ds)
    ds.partitions = (np.arange(n)>=n/2).astype(int)
    ds_train = ds[ds.partitions==part]
    ds_test = ds[ds.partitions!=part]
    if null_model:
        # Separate permuted training targets from true targets used for testing
        ds_train.targets = np.random.permutation(ds_train.targets) # scramble targets
    clf.train(ds_train)    
    pred = clf.predict(ds_test)
    #method to explicitly derive SVM predictions from margin-distance (psuedo probability) estimates
    #get_pred=lambda pred: sorted(pred.keys())[np.argmax((np.array([pred[k] for k in sorted(pred.keys())]).reshape(-1,len(pred.keys())-1)>0).sum(1))]
    # per-target probabilities from multi-class SVM estimates: 
    get_probs=lambda e: (np.array([e[k] for k in sorted(e.keys())]).reshape(-1,len(ds_test.UT)-1)>0).mean(1)
    if len(ds_test.UT)==7: # pch-class non-binary classifier
        #predictions, multi-class probabilities
        est=[get_probs(clf.ca.estimates[i]) for i in range(len(ds_test))]
    elif len(ds_test.UT)==2:
        est=clf.ca.estimates # binary
    else:
        est=[] # probability estimates don't work for pch-height, check get_probs 
    return ds_test.targets, pred, est

def do_subj_classification(ds_masked, subject, task='timbre', cond='a', clf=None, null_model=False, delay=0, dur=1):
    """
    Classify a subject's data
    
    inputs:
     ds_masked - a masked dataset for subject
       subject - subject  - sid00[0-9]{4}    
          task - choose the clf task
     cond - choose the condition h/i/a
           clf - the classifier (LinearCSVMC)
    null_model - Monte-Carlo permutation test [False]    

    outputs:
        dict = {
            'subj' : 'sid00[0-9]{4}'    # subject id
             'res' : [targets, predictions] # list of targets, predictions
              'est' : probability estimates of predictions (all target classes, per trial)
            'task' : which task was performed
       'cond' : which condition in {'h','i'}
              'ut' : unique targets for task
      'null_model' : whether using monte carlo tests [False]
       }    
    """
    tgts, preds, ests = [], [], []
    ds = _encode_task_condition_targets(ds_masked, subject, task, cond, delay, dur) # returns ds_encoded
    for part in [0,1]: # training partitions ordering # TODO : make [1,0] sp testing should be 0,1 which is dataset order
        if 'stim-enc' in task: # stimulus encoding returns voxel time-series and their predictions
            clf = SKLLearnerAdapter(Lasso(alpha=0.2))
            tgt, pred = do_stimulus_encoding(ds, subject, clf, part, null_model)
            est = [] # no probability estimates for stimulus encoding model
        else: # classification
            clf=P.LinearCSVMC(enable_ca=['probabilities']) if clf is None else clf
            if 'X' not in task:
                ds_part = ds
            else: # cross-decode
                ds_part = ds[np.isin(ds.chunks, [1,2,7,8])] if part==0 else ds[np.isin(ds.chunks, [3,4,5,6])]
            tgt, pred, est = _cv_run(ds_part, clf, part, null_model)
        tgts.extend(tgt)
        preds.extend(pred)
        ests.extend(est)
    if 'stim-enc' in task and null_model: # only retain true targets for correct answers, not permuted !
        tgts = []
    else: 
        tgts = np.array(tgts) # TODO swap order of tgts / preds here, to reflect train/test order
    preds= np.array(preds) 
    ests = np.array(ests)
    return {'subj':subject, 'res':[tgts, preds], 'est': ests, 'task':task, 'cond':cond, 'ut':ds.UT, 'null_model':null_model}

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
    fname = opj(path, 'sub-%s'%subject, 'func', 'sub-%s_task-*_run-%02d_space-%s_%s.nii.gz'%(subject, run, space, parcellation))
    #print fname
    fname = glob.glob(fname)[0]
    ds=P.fmri_dataset(fname)
    found = np.where(np.isin(ds.samples,rois))[1]
    return ds[:,found]

def mask_subject_ds(ds, subj, rois):
    """
    Mask a subject's data for given list of rois
    
    inputs:
         ds - the dataset to mask
       subj - sid00[0-9]{4}
       rois - list of rois to merge e.g. [1005, 1035, 2005, 2035]
    
    outputs:
     ds_masked - the masked dataset (data is copied)
    """

    if subj is not None:
        mask = get_subject_mask('%s'%subj, run=1, rois=rois)
        ds_masked=P.fmri_dataset(P.map2nifti(ds), ds.targets, ds.chunks, P.map2nifti(mask))
        P.poly_detrend(ds_masked, polyord=1, chunks_attr='chunks') # in-place
        P.zscore(ds_masked, param_est=('targets', [1,2])) # in-place    
    else:
        ds_masked = ds.copy()
    return ds_masked

def get_autoencoded_subject_ds(ds, subj, rois, ext='auto', AUTOENCDIR='/isi/music/auditoryimagery2/seanfiles'):
    """
    Fetch a subject's autoencoded data for given list of rois
    
    inputs:
         ds - the dataset to mask
       subj - sid00[0-9]{4}
       rois - list of rois to merge e.g. [1005, 1035, 2005, 2035]
        ext - which file extension from {'lh', 'rh', 'lrh', 'auto'} ['auto']
    
    outputs:
     ds_autoencoded - the merged autoencoded dataset
    """
    auto_ext = ext == 'auto'
    if subj is not None: # if not testing
        ae_ds = [] # list of autoencoded rois for subj
        for roi in rois:
            if auto_ext:
                ext = 'lh' if roi < 2000 else 'rh'
            roi = roi if roi < 2000 else roi - 1000
            with open(opj(AUTOENCDIR,'%s/%d/transformed_%s.p'%(subj,roi,ext))) as f:
                ae_ds.append(pickle.load(f).samples) # autoencoded for given rois
            if ae_ds[-1].shape[1]==0:
                raise ValueError('dataset has no samples %s/%d/transformed_%s.p'%(subj,roi,ext))
        ds_autoenc = P.dataset_wizard(samples=P.hstack(ae_ds), targets=ds.targets, chunks=ds.chunks) 
    else: # testing
        ds_autoenc = ds.copy()
    print("** Found Autoencoded BOLD Data **", subj, rois)
    return ds_autoenc

def do_masked_subject_classification(ds, subj, task, cond, rois=[1030,2030], n_null=N_NULL, clf=None, show=False, delay=0, dur=1, autoenc=True):
    """
    The top-level classification entry point.
    Apply mask and do_subj_classification.

    inputs:
  
            ds - a (masked) dataset                                                                                  
       subject - subject  - sid00[0-9]{4}
          task - choose the clf task                                                                          
          cond - condition {'h', 'i', or 'a'} 
          rois - regions of interest to use [1030,2030]
        n_null - how many Monte-Carlo runs to use [N_NULL]
           clf - the classifier              [LinearCSVMC]
         delay - delay target n TRs wrt BOLD [0]
           dur - event-related data duration [1]
       autoenc - whether to use autoencoded BOLD data [True]

    outputs:
          [targets, predictions], [[null_targets1,null_predictions1], ...]
    """
    if task not in tasks:
        raise ValueError("task %s not in tasks"%task)
    clf = P.LinearCSVMC() if clf is None else clf
    if not autoenc: # use freesurfer parcellation
        ds_masked = mask_subject_ds(ds, subj, rois)
    else:                            # get autoencoded data
        ds_masked = get_autoencoded_subject_ds(ds, subj, rois)
    r=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=False, delay=delay, dur=dur)
    res=[r['res'][0],r['res'][1],r['est']]        
    null=[]
    for _ in range(n_null):
        n=do_subj_classification(ds_masked, subj, task, cond, clf=clf, null_model=True, delay=delay, dur=dur)
        null.append([n['res'][0],n['res'][1],n['est']])
    return res, null

def get_result_stats(res, show=True):
    """
    Print and return the mean and ste stats of clf and null models
    inputs:
        res - classifier result array 
             res[0] =[tgts,preds,prob_ests] 
             res[1]=list [[tgts,preds,prob_ests],[tgts,pred,prob_ests], ... * N_NULL] 
       show - whether to print results
    """    
    t, p = np.array(res[0][:2])
    mn = (p==t).mean()
    se = (p==t).std() / np.sqrt(len(p))
    mn0 = np.array([(p0==t0).mean() for t0,p0,e0 in res[1]]).mean()
    se0 = np.array([(p0==t0).std() for t0,p0,e0 in res[1]]).mean() / np.sqrt(len(p))
    # F1-score = 2 * P * R / ( P + R ) per binary class
    ut = np.unique(t)
    f1c = f1_score(t, p, average=None)
    f1 = f1c.mean()
    f1c0 = np.array([f1_score(t0, p0, average=None) for p0,t0,e0 in res[1]]).mean(0)
    f10 = f1c0.mean()
    stats = {'mn':mn, 'se':se, 'mn0':mn0, 'se0':se0, 'ut':ut, 'f1c':f1, 'f1':np.array(f1).mean(), 'f1c0':f1c0, 'f10':f10}
    if show:
        print("mn: %5.3f se: %5.3f mn0: %5.3f se0: %5.3f f1: %5.3f f10: %5.3f"%(stats['mn'],stats['se'],stats['mn0'],stats['se0'],stats['f1'],stats['f10']))
    return stats

def ttest_result_baseline(subj_res, task, roi, hemi, cond):
    """
    Perform group statistical testing on subjects' predictions against baseline and null model conditions

    inputs:
     subj_res - the raw results (targets,predictions) for each subject, task, roi, hemi, and cond
         task - which task to generate group result for
          roi - which region of interest to use
         hemi - which hemisphere [0, 1000] -> L,R
         cond - heard: 'h' or imagined: 'i'

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
    stats=[]
    if 'stim-enc' not in task:
        hemiL = 'LH' if not hemi else 'RH'    
        for subj in subj_res.keys(): # subjects
            r=subj_res[subj][task][roi][hemiL][cond]
            res.append([r[0][0], r[0][1]])
            s=get_result_stats(r, show=False)
            stats.append([s['f1'],s['f10'],s['mn'],s['mn0']]) # collect clf statistics: f1, f10, mn, mn0
        res=np.array(res)
        stats=np.array(stats).mean(0)
        a = (res[:,0,:]==res[:,1,:]).mean(1) # list of WSC mean accuracy 
        ae = a.std() / np.sqrt(len(a))      # SE of mean accuracy 
        am =  a.mean() # overall WSC mean accuracy 
        bm=0.0
        be=0.0
        ut = np.unique(r[0][0]) # targets
        bl = 1.0 / len(ut)
        tt = P.ttest_1samp(a, bl , alternative='greater') # pymvpa's ttest_1samp
        wx = wilcoxon(a-bl) # non-parametric version of ttest_1samp
        #print "TT:(%4.1f, %0.6f)"%(tt[0],tt[1]),"WX:(%4.1f, %0.6f)"%(wx[0],wx[1])
    else: # FIX ME - new stimenc returns tgts and preds
        hemiL = 'LH' if not hemi else 'RH'    
        for subj in subj_res.keys(): # subjects
            r=subj_res[subj][task][roi][hemiL][cond]
            res.append([r[0][0].mean(), r[1][0].mean()]) # FIX ME
        res=np.array(res)
        tt = ttest_rel(res[:,0],res[:,1]) # model vs null
        wx = wilcoxon(res[:,0],res[:,1]) # model vs null
        am = res[:,0].mean()
        ae = res[:,0].std() / np.sqrt(len(res))
        bm = res[:,1].mean()
        be = res[:,1].std() / np.sqrt(len(res))
        ut = np.array([0,2,4,-1,1,3,5]) # pcs as 5ths
    r_res = {'tt':tt, 'wx':wx, 'mn':am, 'se':ae, 'mn0':bm, 'se0':be, 'ut': ut}
    if len(stats==4):
        r_res.update({'f1':stats[0],'f10':stats[1],'mn':stats[2],'mn0':stats[3]})
    return r_res

def ttest_result_null(subj_res, task, roi, hemi, cond):
    """
    Perform group statistical testing on subjects' predictions against null models

    inputs:
     subj_res - the raw results (targets,predictions) for each subject, task, roi, hemi, and cond
         task - which task to generate group result for
          roi - which region of interest to use
         hemi - which hemisphere [0, 1000] -> L,R
         cond - heard: 'h' or imagined: 'i'

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
    stats=[]
    if 'stim-enc' not in task:
        res=[]
        null=[]
        hemiL = 'LH' if not hemi else 'RH'
        for subj in subj_res.keys(): # subjects
            r=subj_res[subj][task][roi][hemiL][cond]
            res.append([r[0][0], r[0][1]])
            null.append( np.array([np.array(r[0][0]==n[1]).mean() for n in subj_res[subj][task][roi][hemiL][cond][1]]).mean() )
            s=get_result_stats(r, show=False)
            stats.append([s['f1'],s['f10'],s['mn'],s['mn0']]) # collect clf statistics: f1, f10, mn, mn0
        stats=np.array(stats).mean(0)
        res=np.array(res)
        a = (res[:,0,:]==res[:,1,:]).mean(1) # list of WSC mean accuracy 
        ae = a.std() / np.sqrt(len(a))      # SE of mean accuracy 
        am =  a.mean() # overall WSC mean accuracy 
        null=np.array(null)
        b = null # list of null WSC null mean accuracy
        be = b.std() / np.sqrt(len(b)) # SE of mean        
        bm= b.mean() # overall WSC null mean accuracy 
        ut = np.unique(r[0][0]) # targets
        #bl = 1.0 / len(ut)
        tt = ttest_rel(a, b) 
        wx = wilcoxon(a,b) # non-parametric version
        #print "TT:(%4.1f, %0.6f)"%(tt[0],tt[1]),"WX:(%4.1f, %0.6f)"%(wx[0],wx[1])
    else: # FIX ME - stim-enc now returns tgts and preds
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
    r_res = {'tt':tt, 'wx':wx, 'mn':am, 'se':ae, 'mn0':bm, 'se0':be, 'ut': ut}
    if len(stats==4):
        r_res.update({'f1':stats[0],'f10':stats[1],'mn':stats[2],'mn0':stats[3]})
    return r_res

def ttest_per_subj_res(subj_res):
    """
    for each subject, perform a 1-sample t-test on the per-target, per-run accuracies against the NULL
    return t-test result dict
    """
    rois = get_LH_roi_keys()
    res={}
    for subj in subjects:
        res[subj]={}
        for task in tasks:
            if 'stim-enc' not in task:
                res[subj][task]={}            
                for roi in rois:
                    res[subj][task][roi]={}
                    for hemi in ['LH','RH']:
                        res[subj][task][roi][hemi]={}
                        for cond in ['h','i']:
                            r=subj_res[subj][task][roi][hemi][cond][0]
                            x = (r[0]==r[1]).mean()
                            ut = np.unique(r[0])
                            y = np.array([(r[0]==n[1]).mean() for n in subj_res[subj][task][roi][hemi][cond][1]]) 
                            tt=P.ttest_1samp(np.tile(y,10),x,alternative='less') # null test, repeated data 
                            wx=wilcoxon(np.tile(y,10)-x) # null test, repeated data
                            res[subj][task][roi][hemi][cond]={'mn':x, 'se':0,
                                                    'm0':y.mean(), 'se0':y.std()/np.sqrt(len(y)),
                                                          'ut':ut, 'tt':tt,'wx':wx, 'baseline': 1.0 / len(ut)}
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

def calc_group_results(subj_res, null_model=True, bilateral=False):
    """
    Calculate all-subject group results for tasks, rois, hemis, and conds
    Ttest and wilcoxon made relative to baseline of task

    inputs:
          subj_res - per-subject raw results (targets, predictions) per task,roi,hemi, and cond
        null_model - whether to use null model, else use baseline model [True]
         bilateral - whether the result dataset is bilateral {LH -> LH+RH only} [False]
    outputs:
       group_res - group-level ttest / wilcoxon results over within-subject means
    """
    group_res = {}
    subjects=subj_res.keys()
    if len(subjects)<2:
        print "Warning: *** Too Few Subjects for Group Analysis, Performing Anyway.... ****"
    hemi_l = [0] if bilateral else [0, 1000]
    for task in subj_res[subjects[0]].keys():
        group_res[task]={}
        for roi in subj_res[subjects[0]][task].keys():
            group_res[task][roi]={}
            for hemi in hemi_l:
                hemiL = 'LH' if not hemi else 'RH'
                group_res[task][roi][hemiL]={}
                for cond in ['h','i']:
                    #print task, roi_map[roi+hemi].replace('ctx-',''), cond.upper(),
                    if null_model:
                        group_res[task][roi][hemiL][cond] = ttest_result_null(subj_res, task, roi, hemi, cond)
                    else:
                        group_res[task][roi][hemiL][cond] = ttest_result_baseline(subj_res, task, roi, hemi, cond)
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
        if 'stim-enc' in task:
            if 'baseline' in r:
                bl = r['baseline']
            else:
                bl = r['mn0']
        else:
            bl = 1.0 / len(r['ut'])
        ax.set_ylim(min(bl,np.array(mins).min())*0.95,mx*1.05)
        pl.xticks((np.arange(len(xlabs))+0.5)*dp,xlabs, rotation=90, fontsize=16)
        if 'stim-enc' in task:
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
                        p = r['tt'][1] # ttest 
                        stars = _get_stars(r['mn'], bl, p, 'stim-enc' in task)
                        pl.text(pos-0.666*len(stars), (r['mn']+r['se'])+rnge*0.05, stars, color='k', fontsize=12)
                    if not np.isnan(r['wx'][0]) and r['tt'][0]>0: # wx is not signed, so use tt for effect sign
                        p = r['wx'][1] #  wilcoxon 
                        stars = _get_stars(r['mn'], bl, p, 'stim-enc' in task)
                        pl.text(pos-0.666*len(stars), (r['mn']+r['se'])+rnge*0.075, stars, color='r', fontsize=12)
                    pos+=dp

def save_result_subj_task(res, subject, task, resultdir=None):
    """
    Save partial (subj,task) results data from a classifier
    
    inputs:
        res  - results output of do_subj_classification 
     subject - subject id - sid00[0-9]{4}
      task   - name of task from tasks

    outputs:
        saves file in OUTDIR "%s_%s_res_part.pickle"%(subject,task)
    """
    resultdir = OUTDIR if resultdir is None else resultdir
    fname = "%s_%s_res_part.pickle"
    with open(opj(resultdir, fname%(subject,task)), "w") as f:
        pickle.dump(res, f)

def load_all_subj_res_from_parts(tsks=tasks, subjs=subjects, resultdir=None):
    """
    Load all partial result files and concatenate into a single dict
    
    inputs: 
       tsks - list of tasks [tasks]
      subjs - list of subjects [subjects]
    outputs:
       subj_res - per-subject results dict, indexed by sid00[0-9]{4}
    """
    resultdir = OUTDIR if resultdir is None else resultdir
    subj_res={}
    for subj in subjs:
        subj_res[subj]={}
        for task in tsks:
            fname = "%s_%s_res_part.pickle"
            with open(opj(resultdir, fname%(subj,task)), "r") as f:
                res_part = pickle.load(f)
                subj_res[subj].update(res_part[subj])
    return subj_res

def export_res_csv(subj_res=None, subj_tt=None, group_tt=None, integrity_check=True):
    """
    Save results dataset(s) as csv file
    inputs:
        subj_res  - subject-level predictions
        subj_tt - subject-level statistical tests
        group_tt - group-level statistical tests
    outputs:
        csv file to: OUTDIR/audimg_subj_res.csv
    """
    swap_timbres = np.array([1,0,3,2])
    join_runs = np.array([2,3,2,3,0,1,0,1]) # swap train/test and interleave h and i conditions
    fname = "audimg_subj_res_concat_cols.csv"
    with open(opj(OUTDIR, fname), "wt") as f:
        writer = csv.writer(f)
        if subj_res is None:
            print("subj_res...")
            subj_res = load_all_subj_res_from_parts()
        if subj_tt is None:
            print("subj_tt...")            
            subj_tt = ttest_per_subj_res(subj_res)
        if group_tt is None:
            print("group_tt...")
            group_tt = calc_group_results(subj_res, null_model=True)
        # subj, task, roi, hemi, cond
        rois = get_LH_roi_keys()
        db = []
        runs, tgts=4, 42
        first_row = True
        # header
        for subj in subjects:            
            for runi, cond in enumerate(['h','h','i','i','h','h','i','i']): # fMRI experiment trial order
                run_slc2 = slice(join_runs[runi]*tgts,(join_runs[runi]+1)*tgts,2) # report targets only once, skip alternate                
                for tgti, tgt in enumerate(range(tgts*runs)[run_slc2]): # skip alternate targets to undo 2 x TR
                    if first_row:
                        rowh=['Trial_Key']
                    row = ["%s.%d.%d"%(subj[-4:],runi,tgti)]
                    for task in ["pch-class","pch-classX","timbre","timbreX"]: # subj_res[subjects[0]].keys(): # may be short a key
                        for roi in rois:
                            for hemidx, hemi in enumerate(['LH','RH']):
                                prefix="%d_%s_"%(roi+hemidx*1000, task)
                                if first_row:
                                    rowh.extend([prefix+"pred1",prefix+"pred2",prefix+"targ",prefix+"run_mn",prefix+"sub_mn"])
                                r = subj_res[subj][task][roi][hemi][cond][0] # true-model trials (tgt, pred, probs)
                                n = subj_res[subj][task][roi][hemi][cond][1] # true-model trials (tgt, pred, probs)                            
                                t = subj_tt[subj][task][roi][hemi][cond]
                                if legend[accessions[subj]][0]=='HC': # undo run reordering due to reverse timbre conditions, only on 1st of each pair
                                    r_orig, n_orig = r, n
                                    r, n = [], []
                                    for i, rr in enumerate(r_orig):
                                        s = rr.shape
                                        r.append(rr.reshape(runs,-1)[swap_timbres,:].reshape(s)) # reverse order of T and C, careful with probs
                                    for i, nn in enumerate(n_orig):
                                        n.append([nnn.reshape(runs,-1)[swap_timbres,:].flatten() for nnn in nn]) # reverse order of T and C                                
                                run_slc = slice(join_runs[runi]*tgts,(join_runs[runi]+1)*tgts,1) # calc run mean
                                run_mn = (r[0][run_slc]==r[1][run_slc]).mean()
                                row.extend([int(r[1][tgt]), int(r[1][tgt+1]), r[0][tgt], run_mn, t['mn']])
                    if first_row:
                        db.append(rowh)                                            
                        first_row=False                                    
                    db.append(row)                    
                    if len(row)!=len(db[0]):
                        raise ValueError("row is incorrect length %d != %d"%(len(row),len(db[0])))                    
        writer.writerows(db)
    if integrity_check:
        with open(opj("export", "ImagAud_Database_V1.csv"), "rt") as f:
            reader = csv.reader(f)
            db_check = [line for line in reader]        
        return db, db_check
    else:
        return True

def export_res_nifti(grp_res, task='pch-class', cond='h', measure='mn', ref_subj=subjects[0]):
    """
    Export group results as nifti file
    """
    fname = opj(DATADIR, 'sub-%s'%ref_subj, 'func', 'sub-%s_task-*_run-%02d_space-%s_%s.nii.gz'%(ref_subj,1,MRISPACE, PARCELLATION))
    ds= P.fmri_dataset(glob.glob(fname)[0]) # refence subject T2w MRISPACE PARCELLATION, could be done with T1w image?
    ds_res = ds.copy(deep=True)
    ds_res.samples[:]=0
    for roi in grp_res[task]:
        for h, hemi in enumerate(['LH','RH']):
            if cond=='i-h':
                value = int( 1000 * ( grp_res[task][roi][hemi]['i'][measure] - grp_res[task][roi][hemi]['h'][measure] ) )
            elif cond=='h-i':
                value = int( 1000 * ( grp_res[task][roi][hemi]['h'][measure] - grp_res[task][roi][hemi]['i'][measure] ) )
            else: 
                value = int(grp_res[task][roi][hemi][cond][measure]*1000)
            ds_res.samples[:,np.where(ds.samples==roi+h*1000)[1]]=value
            print roi, hemi, value
            #    if np.any(ds_res.samples<0):
            #        ds_res.samples = ds_res.samples - ds_res.samples.min() # normalize location
    ds_res_ni=P.map2nifti(ds_res, ds_res.samples)
    ds_res_ni.to_filename('all_grp_res_%s_%s_%s.nii.gz'%(task, cond, measure))
    return True

def roi_analysis(grp_res, h=True, i=True, t=0.05, task='pch-class', full_report=True, bilateral=False, tt='tt', ftxt=None):
    # Report rois shared / not shared between conditions
    if task[-1].lower()=='x':
        htask = task[:-1]
    else:
        htask = task
    itask = task
    hemi_l = ['LH'] if bilateral else ['LH', 'RH']
    f1l=[]
    if True:
        for r in get_LH_roi_keys():
            short_report = ''
            for hemi in hemi_l:
                cond = 'h'
                p=grp_res[htask][r][hemi][cond][tt][1]
                m=grp_res[htask][r][hemi][cond]['mn']
                f1=grp_res[htask][r][hemi][cond]['f1']
                f10=grp_res[htask][r][hemi][cond]['f10']
                cond = 'i'
                pi=grp_res[itask][r][hemi][cond][tt][1]
                mi=grp_res[itask][r][hemi][cond]['mn']            
                f1i=grp_res[htask][r][hemi][cond]['f1']
                f10i=grp_res[htask][r][hemi][cond]['f10']
                #f1i=grp_res[htask][r][hemi][cond]['f1']
                #f1c0i=grp_res[htask][r][hemi][cond]['f1c0']
                f1l.extend([f1,f1i])
                report = None
                if (h and i) and (p<=t and pi<=t):
                    report='h&i'            
                elif (not h and p>t) and (i and pi<=t):
                    report='!h&i'
                elif (h and p<=t) and (not i and pi>t):
                    report='h&!i'
                elif (not h and p>t) and (not i and pi>t):
                    report='!h&!i'
                if report is not None:
                    if full_report:
                        s = "%d %24s %s %s %10s %5.3f (p=%5.3f) f1=%5.3f (f1'=%5.3f) %10s %5.3f (p=%5.3f) f1=%5.3f (f1'=%5.3f)"%(r, roi_map[r].replace('ctx-lh-',''), hemi, report, htask, m, p, f1, f10, itask, mi, pi, f1i, f10i)
                        print (s)
                        if ftxt is not None:
                            ftxt.write(s+'\n')
                    else:
                        if short_report=='':
                            short_report = "%d %24s %s "%(r, roi_map[r].replace('ctx-lh-',''), report)
                        short_report += hemi+' '
            if not full_report and short_report != '':
                print(short_report)
                if ftxt is not None:
                    ftxt.write(short_report+'\n')
    #print("f1: ", sorted(f1l)[::-1][:20]) # top-20 f1-scores
                    
if __name__=="__main__":
    """
    Classify subject BOLD data using task for all ROIs and save subject's results
    Usage: python audimg sid00[0-9]{4} task{pch-height|pch-class|pch-hilo|timbre}    
    """
    arg = 0
    if len(sys.argv) < 3:
        print "Usage: %s sid00[0-9]{4} task{pch-height|pch-class|pch-hilo|timbre} [delay(int) dur(int) n_null(int)]"%sys.argv[arg]
        sys.exit(1)

    arg += 1
    subj = sys.argv[arg] 
    if subj not in subjects:
        print "%s not in subjects"%subj
        print "subjects:", subjects
        sys.exit(1)        
    print("subj: %s"%subj)

    arg += 1
    task = sys.argv[arg] 
    if task not in tasks:
        print "%s not in tasks"%task
        print "tasks:", tasks
        sys.exit(1)        
    print("task: %s"%task)

    arg += 1
    delay = 0 # if > 0, then delay targets relative to BOLD signal (in TRs)
    if len(sys.argv) > arg:
        delay = int(sys.argv[arg]) # delay in TRs relative to BOLD
        print("setting delay = %d"%delay)
        
    arg += 1
    dur = 1 # if > 1, then form event-related dataset of this duration (in TRs) 
    if len(sys.argv) > arg:
        dur = int(sys.argv[arg])   # form event-related dataset of duration in TRs 
        print("setting duration = %d"%dur)

    arg += 1
    n_null = N_NULL # compute null model by deault, use N_NULL models
    if len(sys.argv) > arg:
        n_null = int(sys.argv[arg]) # 0=False, >0=num null models
        print("setting n_null = %d"%n_null)

    arg += 1
    autoenc = 0
    if len(sys.argv) > arg:
        autoenc = int(sys.argv[arg]) # 0=False, 1=True, 2=bilateral
        print("setting autoenc = %d"%autoenc)

    hemi_l = [0] if autoenc==2 else [0, 1000] # 2==bilateral

    # Cortical regions of interest, group_results are L-R lateralized with R=roi_id + 1000
    rois = get_LH_roi_keys()
    ds=get_subject_ds(subj)
    res={}
    res[subj]={}
    res[subj][task]={}
    for roi in rois:
        res[subj][task][roi]={}
        for hemi in hemi_l:                          
            hemiL = 'LH' if not hemi else 'RH'
            res[subj][task][roi][hemiL]={}            
            for cond in ['h','i']:
                res[subj][task][roi][hemiL][cond]=do_masked_subject_classification(ds, subj, task, cond, [roi+hemi], n_null=n_null, delay=delay, dur=dur, autoenc=autoenc)
    save_result_subj_task(res, subj, task)        
