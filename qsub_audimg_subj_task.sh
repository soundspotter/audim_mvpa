#!/bin/bash
export subj
export task

for subj in sid001401 sid001419 sid001410 sid001541 sid001427 sid001088 sid001581 sid001571 sid001660 sid001661 sid001664 sid001665 sid001125 sid001668 sid001672 sid001678 sid001680
do
    for task in pch-height pch-class pch-hilo timbre pch-helix-stim-enc pch-classX timbreX; 
    do
	# Do not overwrite results, remove them if recompute required
	if [ ! -e results_audimg_subj_task_mkc/${subj}_${task}_res_part.pickle ];
	then
	    echo ${subj} ${task}
	    mksub -V run_audimg_subj_task.qsub
	    sleep 1 # prevent race conditions on queue memory hang
	fi
    done
done

