#!/bin/bash
export subj
export task
export delay=0
export dur=1
export nnull=100
export autoenc=1

for subj in sid001680 # sid001401 sid001419 sid001410 sid001541 sid001427 sid001088 sid001581 sid001571 sid001660 sid001661 sid001664 sid001665 sid001125 sid001668 sid001672 sid001678 sid001680
do
    for task in pch-class; #timbre pch-classX timbreX pch-height ; # pch-hilo pch-helix-stim-enc ; 
    do
	echo subj=${subj} task=${task} delay=${delay} dur=${dur} autoenc=${autoenc}
	mksub -V run_audimg_subj_task.qsub
	sleep 1 # prevent race conditions on queue memory hang
    done
done

