#!/bin/bash
#ls -d am2/data/bids3/Casey/Casey/1058_auditoryimagery/sub-sid00* | while read l;
for l in sub-sid001088; #sub-sid001427 sub-sid001541 sub-sid001564 sub-sid001581 sub-sid001594; 
do
    subj=`echo ${l/*\/}`;
    echo qsub -v subj=$subj run_fmriprep.qsub
    qsub -v subj=$subj run_fmriprep.qsub
done

