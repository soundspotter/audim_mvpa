#!/bin/bash
export subj
export task
export null

null=1
null_str="_null"

for subj in sid001401 sid001419 sid001410 sid001541 sid001427 sid001088 sid001581 sid001571 sid001660 sid001661 sid001664 sid001665 sid001125 sid001668 sid001672 sid001678 sid001680
do
    for task in pch-height pch-class pch-hilo timbre; # pch-helix-stim-enc; # stim-enc already compares to NULL
    do
	# Do not overwrite results, remove them if recompute required
	if [ ! -e results_audimg_subj_task/${subj}_${task}_res_part${null_str}.pickle ];
	then
	    echo ${subj} ${task} ${null_str}
	    mksub -V run_audimg_subj_task.qsub
	    sleep 1 # prevent race conditions on queue memory hang
	fi
    done
done

