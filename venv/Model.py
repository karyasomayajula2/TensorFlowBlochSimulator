import tensorflow as tf
import scipy as sp
import numpy as np
import data as d
import math as m
import matplotlib.pyplot as plt

#Preprocess Data
inputData = d.data.getSheppLogan(0);
Nrep = 32;
Nactions = 32;
PDArray = inputData;
PD = tf.constant(PDArray, dtype=tf.complex64, shape=(Nrep, Nactions));
PDvec = tf.reshape(PD, [Nrep*Nactions, 1]);
#PDNormalized = PD/250;
T1 = 100.0; #constant for now
T2 = 20.0; #constant for now

xVec = tf.cast(tf.linspace(1, 32, 32), tf.complex64);
yVec = tf.cast(tf.linspace(1, 32, 32), tf.complex64);

alpha = tf.Variable(tf.zeros([Nrep, Nactions],  dtype= tf.complex64));
deltat = tf.Variable(tf.ones([Nrep, Nactions], dtype = tf.complex64));
gradientX = tf.Variable(tf.zeros([Nrep, Nactions],  dtype= tf.complex64));
gradientY = tf.Variable(tf.zeros([Nrep, Nactions],  dtype= tf.complex64));

## INCORRECT HAVE TWO GRADIENT GX AND GY
## INITIALIZE ALPHA AND GRADIENTS TO 0
## ? INITIALIZE DELTAT to == ? Real #'s
#Bloch Simulator Functions

def xrot(phi):
    zero = tf.constant(0, dtype = tf.complex64)
    one = tf.constant(1, dtype = tf.complex64)
    a = tf.stack([one, zero, zero, zero, tf.cos(phi), -tf.sin(phi), zero, tf.sin(phi), tf.cos(phi)])
    RotX = tf.reshape(a, (3,3))
    return RotX;

def yrot(phi):
    zero = tf.constant(0, dtype=tf.complex64)
    one = tf.constant(1, dtype=tf.complex64)
    a = tf.stack([tf.cos(phi), zero, tf.sin(phi), zero, one, zero, -tf.sin(phi), zero, tf.cos(phi)]);
    RotY = tf.reshape(a, (3,3))
    return RotY;

def zrot(phi):
    zero = tf.constant(0, dtype=tf.complex64)
    one = tf.constant(1, dtype=tf.complex64)
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
    zero = tf.constant(0, dtype=tf.complex64)
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
    z = tf.multiply(comp, func);
    precess = tf.exp(z);
    precessLine = tf.reshape(precess, [Nrep*Nactions, 1])
    ex = tf.constant([[1], [0], [0]], dtype = tf.complex64)
    ey = tf.constant([[0], [1], [0]], dtype = tf.complex64)
    #mx = tf.matmul(m, ex);  #### INCORRECT not z magnetization TRANSVERSE MAGNETIZATION IS THE X Y CHOOSE ONE OR MAGNITUDE srt(MX^2+MY^2)
    #my = tf.matmul(m, ey);
    #mx2 = tf.multiply(mx, mx);
    #my2 = tf.multiply(my, my);
    #mtransverse = tf.sqrt(tf.add(mx2, my2)); # sqrt could not be implemented
    mtransverse = tf.matmul(m, ex); #this works for gradients
    #mz = tf.cast(mz, tf.complex64);
    msig = tf.reshape(tf.multiply(mtransverse, precessLine), [Nrep, Nactions]); #check matrix dimensions for this portion
    return msig;

def signal(m, gradientX, gradientY, x, y):
    #x[tf.newaxis, :], y[: , tf.newaxis]]
    svec = tf.reduce_sum(gradprecess(m, gradientX, gradientY, x[tf.newaxis, :], y[: , tf.newaxis]));
    return svec;

def forward(PD, T1, T2, alpha, gradientX, gradientY, deltat, x, y, xVec, yVec, rfPhase=0.0): #Pd is a tensor, T1 and T2 are scalars for that image
    ez = tf.constant([0, 0, 1], dtype=tf.complex64, shape=(1, 3))
    #s = tf.Variable(tf.zeros([Nrep, Nactions], dtype=tf.complex64));
    s = [];
    m0 = tf.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    t1 = tf.constant(T1, dtype = tf.complex64)
    t2 = tf.constant(T2, dtype = tf.complex64)
    phase = tf.constant(rfPhase, dtype=tf.complex64)
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r,a], phase); #no .numpy() anywhere
            m = tf.matmul(m, flip) # 64by3 times 3by3 = 64by3
            rel = relax(deltat[r,a], t1, t2);
            m = tf.add(tf.matmul(m, rel), tf.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            #b = freeprecess(deltat[r,a]);  ## larmor frequency times deltat t
            #m = tf.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            #ms = gradprecess(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, x, y) #64 by 1 vector transverse magnetiztion
            sIndex = signal(m, gradientX[r,a], gradientY[r,a], xVec, yVec);
            s.append(sIndex); ## TAKE REAL COMPONENT OF SIGNAL TO RECONSTRUCT THE IMGE
    X = tf.stack(s);
    Y = tf.reshape(X, [x,y]);
    return Y;


def reconstruction(s):
    rec = tf.signal.ifft2d(s);
    #recon = tf.math.real(rec);
    #recon = tf.math.real(rec)+tf.math.imag(rec); #use abs or real
    #recon = tf.cast(recon, tf.float64);
    recon1 = tf.reshape(rec, [Nrep*Nactions,1]);
    return recon1;


def cost(recon, PD):
    #lossNorm = loss/tf.reduce_max(loss);
    cost = tf.reduce_sum(tf.square(PD - recon));
    return cost; ## SUM OF THE SQUARES OF THE DIFFERENCES

opt = tf.keras.optimizers.Adam(learning_rate=0.01);
input = PDvec; #vector 4096 by 1
targetContrast = PDvec; #64 by 64
epochs = 10;
#mainLoss = 100000;
def train(opt, input, targetContrast):
    with tf.GradientTape() as tape:
        vars = [alpha, deltat]; #gradients, deltat];
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

        result = forward(input, T1, T2, alpha, gradientX, gradientY, deltat, Nrep, Nactions, xVec, yVec);
        groundTruthSignal = tf.ones([Nrep*Nactions, 1], dtype = tf.complex64)
        ##Can be computed in the loop and have non nan values
        #mGroundTruth = tf.zeros([Nrep*Nactions, 3], dtype = tf.complex64);
        #loss = tf.reduce_sum(tf.square(mGroundTruth - result));

        #rec = reconstruction(tf.reshape(result);
        #groundSigRec = reconstruction(groundTruthSignal);
        loss = tf.reduce_sum(tf.square(groundTruthSignal - result));
        #loss = cost(rec, targetContrast);

        grads = tape.gradient(loss, vars)
        opt.apply_gradients(zip(grads, vars))
    return rec, loss;

#Change to make it in loss dependent
l = [];
for i in range(0, epochs):
    resArr = train(opt, input, targetContrast);
    plt.figure(1);
    a = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(targetContrast)), [Nrep ,Nactions]), clim = [0,1]);
    plt.figure(2);
    b = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(resArr[0])), [Nrep, Nactions]), clim=[0, 1]);
    plt.gray()
    plt.show()
    l.append(resArr[1]);
x = np.linspace(1, epochs, epochs+1)
plt.plot(x, l)
print(alpha)
print(deltat)
print(gradients)
