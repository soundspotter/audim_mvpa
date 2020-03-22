#!/bin/bash
export subj
export task
for subj in sid001125 sid001571 sid001613 sid001660 sid001661 sid001664 sid001665 sid001668 sid001672 sid001678 sid001679 sid001680 # batch 3
do
    for task in pch-height pch-class pch-hilo timbre pch-helix-stim-enc;
    do
	# Do not overwrite results, remove them if recompute required
	if [ ! -e results_audimg_subj_task/${subj}_${task}_res_part.pickle ];
	then
	    echo mksub -V run_audimg_subj_task.qsub
	    mksub -V run_audimg_subj_task.qsub
	    sleep 1 # prevent race conditions on queue memory hang
	fi
    done
done

