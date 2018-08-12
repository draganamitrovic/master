#!/usr/bin/env python

# integracija MPI biblioteke
from mpi4py import MPI
import sys
import numpy as np

import random

# importovanje ostalih biblioteka
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift
from skimage import data
from skimage.feature import register_translation


totalNumberOfImages = int(sys.argv[1]) #ukupan broj slika koje treba da se obrade

TaskMaster = 0 #redni broj glavnog procesora/čvora

# definicija promenjljivih
image = data.camera() #originalna slika
offset_image = (0, 0)
offsetInImagesSet = [(0, 0)] #definisani pomeraji slike
offsetInImagesCalc = [(0, 0)] #detektovani pomeraji slike

# uvodjenje promenjljivih vezanih za MPI interfejs
comm = MPI.COMM_WORLD
worldSize = comm.Get_size() # ukupan broj procesora
rank = comm.Get_rank() # rang procesora
processorName = MPI.Get_processor_name() # ime procesora

print ("Process %d started.\n" % (rank))
print ("Running from processor %s, rank %d out of %d processors.\n" % (processorName, rank, worldSize))

# izracunavanje broja slika koji ce se obraađivati po procesoru
if (worldSize == 1):
    slice = totalNumberOfImages
else:
    slice = totalNumberOfImages / (worldSize-1)
assert slice >= 1 #provera da li je proces deljiv

############### sinhronizacija procesa
comm.Barrier()

############## ako je rang procesora glavni čvor
if rank == TaskMaster:
    print ("Start")

    for i in range(1, worldSize):
        #slanje broja slika koje treba svaki procesor da obradi
        numberOfImagessPerCPU = slice / worldSize-1
        comm.send(numberOfImagessPerCPU, dest=i, tag=i)

############### sinhronizacija procesa
comm.Barrier()

############### ako je rang procesora worker node
if rank != TaskMaster:

    #primanje broja slika koi treba obraditi
    numberOfImagessPerCPU = comm.recv(source=0, tag=rank)

    print ("Data Received from process %d.\n" % (rank))
    t_start = MPI.Wtime()
    print ("Start Calculation from process %d.\n" % (rank))

    for i in range(0, numberOfImagessPerCPU-1):
        # pomeraj svake slike je nasumičan broj između -100 i 100, po x i y osi
        shift = (random.randint(-100, 100), random.randint(-100, 100))
        offset_image = [(0, 0)]
        offset_image = fourier_shift(np.fft.fftn(image), shift)
        offset_image = np.fft.ifftn(offset_image)

        shift_detected, error, diffphase = register_translation(offset_image, image)

        # niz setovanih i detektovanih pomeraja koji se salju
        sendDataSet = shift
        sendDataCalc = shift_detected
        
    t_diff = MPI.Wtime() - t_start
    print("Process %d finished in %5.4fs.\n" % (rank, t_diff))
    # Send data
    print ("Sending results to Master %d bytes.\n" % (send.nbytes))
    comm.send(sendDataSet, dest=0, tag=rank)
    comm.send(sendDataCalc, dest=0, tag=rank)

comm.Barrier()
#########################################################
numberOfImagessPerCPU = slice / worldSize-1
if rank == TaskMaster:
    print ("Checking response from Workers.\n")
    for i in range(0, numberOfImagessPerCPU):
        recvDataSet = comm.recv(source=i, tag=i)
        offsetInImagesSet[numberOfImagessPerCPU*tag+i] = recvDataSet
        print ("Received response from %d.\n" % (i))

    for j in range(0, numberOfImagessPerCPU):
        recvDataCalc = comm.recv(source=i, tag=i)
        offsetInImagesCalc[numberOfImagessPerCPU*tag+i] = recvDataCalc
        print ("Received response from %d.\n" % (i))

 
    print ("End")
    print ("Offset seted: \n", offsetInImagesSet)
    print ("Offset calculated: \n", offsetInImagesCalc)

comm.Barrier()
