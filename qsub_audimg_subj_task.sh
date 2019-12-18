#!/bin/bash
for subj in sid001401 sid000388 sid001415 sid001419 sid001410 sid001541 sid001427 sid001088 sid001564 sid001581 sid001594;
do
    for task in pch-height pch-class pch-hilo timbre pch-helix-stim-enc;
    do
	# Do not overwrite results, remove them if recompute required
	if [ ! -e results_audimg_subj_task/${subj}_${task}_res_part.pickle ];
	then
	    echo qsub -v subj=$subj,task=$task run_audimg_subj_task.qsub
	    qsub -v subj=$subj,task=$task run_audimg_subj_task.qsub
	    sleep 1 # prevent race conditions on queue memory hang
	fi
    done
done

