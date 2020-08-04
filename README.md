# audim_mvpa
decoding auditory imagery from fmri

1. Setting up your environment

Make sure your environment has the following conda packages installed, in addition to standard Anaconda install (Python 2.x):
    pymvpa pybids

cd ${YourExperimentDirectory}
git clone https://github.com/soundspotter/audim_mvpa.git
ln -s audim_mvpa/audimg.py .
ln -s audim_mvpa/qsub_audimg_subj_task.sh .
ln -s audim_mvpa/run_audimg_subj_task.qsub .

2. How to run classifier jobs on the queue
cd ${YourExperimentDirectory}
EDIT run_audimg_subj_task.qsub for your email address (if you want email notifications from discovery cluster queues)
EDIT audimg.py # set AUTOENCDIR to point to the directory containing the auto-encoded BOLD data

Autoencoder classifiers
. qsub_audimg_subj_task.sh # This shell script will launch jobs to train/test classifiers for all subjects and all experiments (pch-class, pch-classX, timbre, timbre-X, pch-height)

Autoencoder results will be written to the following sub-directory in your current working directory:
results_audimg_subj_task_mkc_del0_dur1_n1000_autoenc

Non-autoencoder classifiers
EDIT qsub_audimg_subj_task.sh # set autoenc=0
. qsub_audimg_subj_task.sh # This shell script will launch jobs to train/test classifiers for all subjects and all experiments (pch-class, pch-classX, timbre, timbre-X, pch-height)

Non-autoencoder results will be written to the following sub-directory in your current working directory:
results_audimg_subj_task_mkc_del0_dur1_n1000

3. Seeing the results

cd ${YourExperimentDirectory}
ipython # launch an interactive python shell


In [1]: import audimg as A
In [2] A.set_resultdir_by_params(delay=0, dur=1, n_null=1000, autoenc=True, update=True) # Set resultdirectory to reflect training/testing parameters
In [2]: subj_res, grp_res = A.collate_model_results(save=True, tt='tt', t=0.01)

This will output the following statistical summaries to the terminal, comparing autoencoder and non-autoencoder classifiers (or something like this, depending on your autoencoder params):

*******************************************************************
results_audimg_subj_task_mkc_del0_dur1_n1000_autoenc_bl
*******************************************************************
H 
1034       transversetemporal LH h  pch-class 0.172 (p=0.000)

I 
1011         lateraloccipital RH i  pch-class 0.166 (p=0.004)
1012     lateralorbitofrontal LH i  pch-class 0.162 (p=0.007)
1013                  lingual LH i  pch-class 0.161 (p=0.000)
1019            parsorbitalis LH i  pch-class 0.162 (p=0.004)
1024               precentral LH i  pch-class 0.162 (p=0.001)
1030         superiortemporal LH i  pch-class 0.174 (p=0.000)
1030         superiortemporal RH i  pch-class 0.162 (p=0.009)
1031            supramarginal LH i  pch-class 0.164 (p=0.009)
1031            supramarginal RH i  pch-class 0.163 (p=0.002)

I X
1011         lateraloccipital RH i pch-classX 0.159 (p=0.008)
1012     lateralorbitofrontal RH i pch-classX 0.159 (p=0.001)

*******************************************************************
results_audimg_subj_task_mkc_del0_dur1_n1000_autoenc_null
*******************************************************************
H 
1034       transversetemporal LH h  pch-class 0.172 (p=0.000)

I 
1011         lateraloccipital RH i  pch-class 0.166 (p=0.008)
1013                  lingual LH i  pch-class 0.161 (p=0.000)
1019            parsorbitalis LH i  pch-class 0.162 (p=0.009)
1024               precentral LH i  pch-class 0.162 (p=0.002)
1030         superiortemporal LH i  pch-class 0.174 (p=0.000)
1031            supramarginal RH i  pch-class 0.163 (p=0.004)

I X
1012     lateralorbitofrontal RH i pch-classX 0.159 (p=0.002)

*******************************************************************
results_audimg_subj_task_mkc_del0_dur1_n1000_bl
*******************************************************************
H 
1001                 bankssts RH h  pch-class 0.176 (p=0.001)

I 
1018          parsopercularis RH i  pch-class 0.167 (p=0.010)
1020         parstriangularis RH i  pch-class 0.164 (p=0.005)

I X
1027     rostralmiddlefrontal RH i pch-classX 0.163 (p=0.003)
1034       transversetemporal LH i pch-classX 0.164 (p=0.005)

*******************************************************************
results_audimg_subj_task_mkc_del0_dur1_n1000_null
*******************************************************************
H 
1001                 bankssts RH h  pch-class 0.176 (p=0.001)

I 

I X
1027     rostralmiddlefrontal RH i pch-classX 0.163 (p=0.006)


_bl = baseline model evaluation, _null = null model evaluation



