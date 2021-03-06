import tensorflow as tf
import scipy as sp
import numpy as np
import data as d
import math as m
import matplotlib.pyplot as plt

#Preprocess Data
inputData = d.data.getSheppLogan(0);
Nrep = inputData.shape[0];
Nactions = inputData.shape[1];
PDArray = inputData;
PD = tf.constant(PDArray, dtype=tf.float64, shape=(Nrep, Nactions));
PDvec = tf.reshape(PD, [Nrep*Nactions, 1]);
#PDNormalized = PD/250;
T1 = 5.0; #constant for now
T2 = 20.0; #constant for now

xMatrix = tf.cast(tf.linspace(1, 32, 32), tf.float64);
yMatrix = tf.cast(tf.linspace(1, 32, 32), tf.float64);
#xMatrix = tf.stack([xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec,xVec,xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, xVec, ])
#yMatrix = tf.transpose(xMatrix)

alpha = tf.Variable(tf.ones([Nrep, Nactions],  dtype= tf.float64));
deltat = tf.Variable(tf.ones([Nrep, Nactions], dtype = tf.float64));
#gradientX = tf.Variable(tf.zeros([Nrep, Nactions],  dtype= tf.complex64));
#gradientY = tf.Variable(tf.zeros([Nrep, Nactions],  dtype= tf.complex64));
gradientX = tf.Variable([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]], dtype = tf.float64)

                          # gradientX = tf.Variable([[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14 + 0j, 15 + 0j, 16 + 0j, 17 + 0j, 18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26 + 0j, 27 + 0j, 28 + 0j, 29 + 0j, 30 + 0j, 31 + 0j, 32 + 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                         [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j],
#                          [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j, 13 + 0j, 14+0j, 15 + 0j, 16 + 0j, 17 + 0j,  18 + 0j, 19 + 0j, 20 + 0j, 21 + 0j, 22 + 0j, 23 + 0j, 24 + 0j, 25 + 0j, 26+ 0j, 27+ 0j, 28+ 0j, 29+ 0j, 30+ 0j, 31+ 0j, 32+ 0j]], dtype = tf.float64)
gradientY = tf.Variable(tf.transpose(gradientX))

## INCORRECT HAVE TWO GRADIENT GX AND GY
## INITIALIZE ALPHA AND GRADIENTS TO 0
## ? INITIALIZE DELTAT to == ? Real #'s
#Bloch Simulator Functions

def xrot(phi):
    zero = tf.constant(0, dtype = tf.float64)
    one = tf.constant(1, dtype = tf.float64)
    a = tf.stack([one, zero, zero, zero, tf.cos(phi), -tf.sin(phi), zero, tf.sin(phi), tf.cos(phi)])
    RotX = tf.reshape(a, (3,3))
    return RotX;

def yrot(phi):
    zero = tf.constant(0, dtype=tf.float64)
    one = tf.constant(1, dtype=tf.float64)
    a = tf.stack([tf.cos(phi), zero, tf.sin(phi), zero, one, zero, -tf.sin(phi), zero, tf.cos(phi)]);
    RotY = tf.reshape(a, (3,3))
    return RotY;

def zrot(phi):
    zero = tf.constant(0, dtype=tf.float64)
    one = tf.constant(1, dtype=tf.float64)
    a = tf.stack([tf.cos(phi), -tf.sin(phi), zero, tf.sin(phi), tf.cos(phi), zero, zero, zero, one])
    Rotz = tf.reshape(a, (3,3));
    return Rotz;

def throt(phi, phase):
    Rz = zrot(-phase);
    Rx = xrot(phi);
    Rzinv = tf.linalg.inv(Rz);
    Rzinvx = tf.matmul(Rzinv, Rx);
    Rthrot = tf.matmul(Rzinvx, Rz);
    return Rthrot;

def relax(deltat, T1, T2):
    zero = tf.constant(0, dtype=tf.float64)
    exponent1 = -deltat/T2;
    exponent2 = -deltat/T1;
    a = tf.stack([tf.exp(exponent1), zero, zero, zero, tf.exp(exponent1), zero, zero, zero, tf.exp(exponent2)]);
    rel = tf.reshape(a, (3, 3));
    return rel;

def freeprecess(deltat):  # for us relax does what freeprecess should do except the zrotation
    gammaH = 42.58 * 10**6; #gyromagnetic ratio/(2*pi) for 1H in Hz/T
    B0 = 1.5; #T, main field B0 (full-body imaging systems usually 0.1T-1.5T
    phi = gammaH * B0 * deltat*10**(-3); #radians, converted deltat to s. formula from http://mriquestions.com/what-is-flip-angle.html
    #phi = 2 * (phase) * df * deltat / 1000; ##larmor freq times delta t
    # E1 = np.exp(-deltat / T1);
    # E2 = np.exp(-deltat / T2);
    b = zrot(phi);
    # Afp = tf.matmul(tf.constant([[E2, 0, 0], [0, E2, 0], [0, 0, E1]], dtype= tf.float64), b);
    # Bfp = tf.constant([[0], [0], [1 - E1]], dtype=tf.float64)
    return b;

def gradprecess(m, gradientX, gradientY, x, y):
    gx = gradientX;
    gy = gradientY;
    comp = tf.constant([0 + 1j], dtype=tf.complex64); # 1i
    func = tf.add(gx*x, gy*y);
    #exponential = tf.constant([func], dtype=tf.complex64, shape=(1,1));
    #f = tf.cast(func, dtype=tf.complex64);
    funcComplex = tf.cast(func, tf.complex64)
    z = tf.multiply(comp, funcComplex);
    precess = tf.exp(z);
    precessLine = tf.reshape(precess, [Nrep*Nactions, 1])
    ex = tf.constant([[1], [0], [0]], dtype = tf.float64)
    ey = tf.constant([[0], [1], [0]], dtype = tf.float64)
    mx = tf.matmul(m, ex);  #### INCORRECT not z magnetization TRANSVERSE MAGNETIZATION IS THE X Y CHOOSE ONE OR MAGNITUDE srt(MX^2+MY^2)
    my = tf.matmul(m, ey);
    mx2 = tf.multiply(mx, mx);
    my2 = tf.multiply(my, my);
    mtransverse = tf.sqrt(tf.maximum(tf.math.real(tf.add(mx2, my2)), 1e-5)); # sqrt could not be implemented
    #mtransverse = tf.matmul(m, ex); #this works for gradients
    mtransverse = tf.cast(mtransverse, tf.complex64);
    msig = tf.reshape(tf.multiply(mtransverse, precessLine), [Nrep, Nactions]); #check matrix dimensions for this portion
    return msig;

def signal(m, gradientX, gradientY, x, y):
    #x[tf.newaxis, :], y[: , tf.newaxis]]
    svec = tf.reduce_sum(gradprecess(m, gradientX, gradientY, x[tf.newaxis, :], y[: , tf.newaxis]));
    return svec;

def forward(PD, T1, T2, alpha, gradientX, gradientY, deltat, x, y, xMatrix, yMatrix, rfPhase=0.0): #Pd is a tensor, T1 and T2 are scalars for that image
    ez = tf.constant([0, 0, 1], dtype=tf.float64, shape=(1, 3))
    #s = tf.Variable(tf.zeros([Nrep, Nactions], dtype=tf.complex64));
    s = [[0]*Nrep for i in range(Nactions)]
    m0 = tf.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    t1 = tf.constant(T1, dtype = tf.float64)
    t2 = tf.constant(T2, dtype = tf.float64)
    phase = tf.constant(rfPhase, dtype=tf.float64)
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r,a], phase); #no .numpy() anywhere
            m = tf.matmul(m, flip) # 64by3 times 3by3 = 64by3
            rel = relax(deltat[r,a], t1, t2);
            m = tf.add(tf.matmul(m, rel), tf.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r,a]);  ## larmor frequency times deltat t
            m = tf.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            sIndex = signal(m, gradientX[r,a], gradientY[r,a], xMatrix, yMatrix);
            a1 = (gradientX[r,a].numpy())
            a2 = (gradientY[r,a].numpy())
            a1 = int(a1) - 1;
            a2 = int(a2) - 1;
            #code to fix boundary conditions
            if a1 > Nrep - 1:
                a1 = Nrep - 1
            elif a1 < 0:
                a1 = 0;
            elif a2 > Nactions - 1:
                a2 = Nactions - 1;
            elif a2 < 0:
                a2 = 0;

            s[a1][a2] = sIndex; ## TAKE REAL COMPONENT OF SIGNAL TO RECONSTRUCT THE IMGE
    X = tf.stack(s);
    Y = tf.reshape(X, [x,y]);
    return Y;


def reconstruction(s):  ##must choose when to use fftshift instead of just directly ifft2d
    #sShift = tf.signal.fftshift(s);
    #plt.figure(3);
    #a = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(sShift)), [Nrep, Nactions]), clim=[0, 1]);
    #plt.gray();
    #plt.show();
    rec = tf.signal.ifft2d(s); #sShift
    #recon = tf.math.real(rec);
    #recon = tf.math.real(rec)+tf.math.imag(rec); #use abs or real
    #recon = tf.cast(recon, tf.float64);
    recon1 = tf.reshape(rec, [Nrep*Nactions,1]);
    return recon1;


def cost(recon, PD):
    #lossNorm = loss/tf.reduce_max(loss);
    #PDcomplex = tf.cast(PD, tf.complex64)
    cost = tf.reduce_sum(tf.square(PD - tf.cast(tf.math.abs(recon), tf.float64)));
    return cost; ## SUM OF THE SQUARES OF THE DIFFERENCES

opt = tf.keras.optimizers.Adam(learning_rate=0.01);
input = PDvec; #vector 4096 by 1
targetContrast = PDvec; #64 by 64
epochs = 1000;
#mainLoss = 100000;
def train(opt, input, targetContrast):
    with tf.GradientTape() as tape:
        vars = [alpha, deltat, gradientX, gradientY];
        tape.watch(vars);
        ##Proof that alpha gradients compute by themselves and deltat shouldnt be nan
        ##commented out here
        #mGroundTruth = tf.constant([1,1,1], dtype = tf.complex64);
        #t1 = tf.constant(T1, dtype=tf.complex64)
        #t2 = tf.constant(T2, dtype=tf.complex64)
        #ez = tf.constant([0, 0, 1], dtype=tf.complex64, shape=(1, 3))
        #m0 = tf.matmul(input, ez);
        #m00 = tf.reshape(m0[500], (1, 3));
        #m = tf.reshape(m0[500], (1, 3));
        #phase = tf.constant(0.0, dtype=tf.complex64)
        #flip = throt(alpha[0, 0], phase);
        #print(m);
        #m = tf.matmul(m, flip)
        #print(m);
        #rel = relax(deltat[0, 0], t1, t2);
        #m = tf.add(tf.matmul(m, rel), tf.matmul(m00, 1 - rel));  # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
        #print(m)
        #b = freeprecess(deltat[0, 0], phase, df=10);
        #m = tf.matmul(m, b);
        #print(m)
        #loss = tf.reduce_sum(tf.square(mGroundTruth - m));

        result = forward(input, T1, T2, alpha, gradientX, gradientY, deltat, Nrep, Nactions, xMatrix, yMatrix);
        #groundTruthSignal = tf.ones([Nrep*Nactions, 1], dtype = tf.complex64)
        ##Can be computed in the loop and have non nan values
        #mGroundTruth = tf.zeros([Nrep*Nactions, 3], dtype = tf.complex64);
        #loss = tf.reduce_sum(tf.square(mGroundTruth - result));

        rec = reconstruction(result);
        #groundSigRec = reconstruction(groundTruthSignal);
        #loss = tf.reduce_sum(tf.square(groundTruthSignal - result));
        loss = cost(rec, targetContrast);

    grads = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grads, vars))
    return rec, loss;

#Change to make it in loss dependent
l = [];
for i in range(0, epochs):
    resArr = train(opt, input, targetContrast);
    #plt.figure(1);
    #a = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(targetContrast)), [Nrep ,Nactions]), clim = [0,1]);
    #plt.figure(2);
    #b = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(resArr[0])), [Nrep, Nactions]), clim=[0, 1]);
    #plt.gray()
    #plt.show()
    l.append(resArr[1]);
x = np.linspace(1, epochs, epochs+1)
plt.plot(x, l)
print(alpha)
print(deltat)
print(gradients)
