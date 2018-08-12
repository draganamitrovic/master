#!/bin/bash
# @Author: Dragana Mitrovic
# @Description: Phase Correlation for image shift processing
for IMAGES in 108 216 432; do
	for CPU in 4 7 10 13; do
		mpiexec -f machinefile -n $CPU python /home/pi/mpi4py-2.0.0/demo/imagesgift.py $IMAGES >> mpi$CPU.$IMAGES.log
	done			
done
