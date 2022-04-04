#!/bin/bash
export subj
export task
export stem="results_audimg_subj_task_SVDMAP"
export delay=0
export dur=1
export nnull=10000
export autoenc=0
export svdmap=1.00
export overwrite=0

if [ $autoenc -gt 0 ];
then
    autoencstr="_autoenc"
else
    autoencstr=""
fi

for subj in sid001401 sid001419 sid001410 sid001541 sid001427 sid001088 sid001581 sid001571 sid001660 sid001661 sid001664 sid001665 sid001125 sid001668 sid001672 sid001678 sid001680 
do
    for task in pch-class pch-classX; # timbre timbreX pch-height pch-hilo pch-helix-stim-enc ; 
    do
	if [ ! -e ${stem}_del${delay}_dur${dur}_n${nnull}_svd${svdmap}${autoencstr}/${subj}_${task}_res_part.pickle ];
	then
	    echo subj=${subj} task=${task} delay=${delay} dur=${dur} nnull=${nnull} autoenc=${autoenc} svdmap=${svdmap} overwrite=${overwrite}
	    mksub -V run_audimg_subj_task.qsub
	    #echo python audimg.py ${subj} ${task} ${delay} ${dur} ${nnull} ${autoenc} ${svdmap} ${overwrite}
	    sleep 120 # prevent race conditions on queue memory hang
	fi
    done
done

