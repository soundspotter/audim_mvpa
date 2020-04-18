#!/bin/bash
# BIDS3:
#ls -d am2/data/bids3/Casey/Casey/1058_auditoryimagery/sub-sid00* | while read l;
# sub-sid001088; #sub-sid001427 sub-sid001541 sub-sid001564 sub-sid001581 sub-sid001594;
#
# BIDS4
# for l in sub-sid001125 sub-sid001571 sub-sid001613 sub-sid001660 sub-sid001661 sub-sid001664 sub-sid001665 sub-sid001668 sub-sid001672 sub-sid001678 sub-sid001679 sub-sid001680;
# do
#     subj="$l" #`echo ${l/*\/}`;
#     echo mksub -v subj="$subj" run_fmriprep.qsub
#     mksub -v subj="$subj" run_fmriprep.qsub
# done
# Use mksub -t PBS_ARRAYID
mksub -t 4-5 run_fmriprep.qsub
