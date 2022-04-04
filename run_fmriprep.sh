#!/usr/bin/bash
# usage: run_fmriprep sub-sud00[0-9]{4}
# author: Michael A. Casey, 08/27/19

if [ $# -lt 1 ]
then
    echo "Usage: run_fmriprep sub-sud00[0-9]{4}"
    exit 101
else
    SID="$1" # participant-label
fi

AM2PATH=/isi/music/auditoryimagery2
BIDSDIR=am2/data/bids4/Casey/Casey/1058_auditoryimagery # Make sure this is the correct path to subjects
OUTDIR=am2/data/fmriprep_2mm
BIDSSRC=containers/images
FMRIPREP=fmriprep-1.4.1.simg
FS_LICENSE=/optnfs/freesurfer/6.0.0/license.txt

echo singularity run --cleanenv --bind ${FS_LICENSE}:/opt/freesurfer/license.txt ${BIDSSRC}/${FMRIPREP} \
     --fs-license-file /opt/freesurfer/license.txt \
	    ${BIDSDIR} ${OUTDIR} \
	    participant --participant-label ${SID} 

     singularity run --cleanenv --bind ${FS_LICENSE}:/opt/freesurfer/license.txt ${BIDSSRC}/${FMRIPREP} \
     --fs-license-file /opt/freesurfer/license.txt \
	    ${BIDSDIR} ${OUTDIR} \
	    participant --participant-label ${SID} 



