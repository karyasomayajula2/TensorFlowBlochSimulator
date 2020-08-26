import tensorflow as tf
import scipy as sp
import numpy as np
import data as d
import math as m
import matplotlib.pyplot as plt

#Preprocess Data
inputData = d.data.getSheppLogan(0);
Nrep = 64;
Nactions = 64;
PDArray = inputData;
PD = tf.constant(PDArray, dtype=tf.complex64, shape=(Nrep, Nactions));
PDvec = tf.reshape(PD, [Nrep*Nactions, 1]);
#PDNormalized = PD/250;
T1 = 1.0; #constant for now
T2 = 3.0; #constant for now
xVec = tf.cast(tf.linspace(1, 64, 64), tf.complex64);
yVec = tf.cast(tf.linspace(1, 64, 64), tf.complex64);
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

def freeprecess(deltat, phase, df):  # for us relax does what freeprecess should do except the zrotation
    phi = 2 * (phase) * df * deltat / 1000;
    # E1 = np.exp(-deltat / T1);
    # E2 = np.exp(-deltat / T2);
    b = zrot(phi);
    # Afp = tf.matmul(tf.constant([[E2, 0, 0], [0, E2, 0], [0, 0, E1]], dtype= tf.float64), b);
    # Bfp = tf.constant([[0], [0], [1 - E1]], dtype=tf.float64)
    return b;

def gradprecess(m, gradientX, gradientY, deltat, x, y):
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
    mx = tf.matmul(m, ex);  #### INCORRECT not z magnetization TRANSVERSE MAGNETIZATION IS THE X Y CHOOSE ONE OR MAGNITUDE srt(MX^2+MY^2)
    my = tf.matmul(m, ey);
    mx2 = tf.multiply(mx, mx);
    my2 = tf.multiply(my, my);
    mtransverse = tf.sqrt(tf.add(mx2, my2));
    #mz = tf.cast(mz, tf.complex64);
    msig = tf.reshape(tf.multiply(mtransverse, precessLine), [Nrep, Nactions]); #check matrix dimensions for this portion
    return msig;

def signal(m, gradientX, gradientY, deltat, x, y):
    #x[tf.newaxis, :], y[: , tf.newaxis]]
    svec = tf.reduce_sum(gradprecess(m, gradientX, gradientY, deltat, x[tf.newaxis, :], y[: , tf.newaxis]));
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
            m = tf.matmul(m, flip) # 64by3 times 3by3 = 64by3y
            rel = relax(deltat[r,a], t1, t2);
            m = tf.add(tf.matmul(m, rel), tf.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r,a], phase, df=10);
            m = tf.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            #ms = gradprecess(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, x, y) #64 by 1 vector transverse magnetiztion
            sIndex = signal(m, gradientX[r,a], gradientY[r,a], deltat[r,a], xVec, yVec);
            s.append(sIndex); ## TAKE REAL COMPONENT OF SIGNAL TO RECONSTRUCT THE IMGE
    X = tf.stack(s);
    Y = tf.reshape(X, [x,y]);
    return Y;



#s = forward(PD, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec);
#print(s);

def reconstruction(s):
    rec = tf.signal.ifft2d(s);
    recon = tf.math.real(rec);
    #recon = tf.math.real(rec)+tf.math.imag(rec); #use abs or real
    #recon = tf.cast(recon, tf.float64);
    recon1 = tf.reshape(recon, [Nrep*Nactions,1]);
    return recon1;


def cost(recon, PD):
    loss = tf.square(tf.reshape(tf.math.real(PD), [Nrep*Nactions, 1]) - recon);
    lossNorm = loss/tf.reduce_max(loss);
    cost = tf.reduce_sum(lossNorm);
    return cost; ## SUM OF THE SQUARES OF THE DIFFERENCES

opt = tf.keras.optimizers.Adam(learning_rate=0.001);
input = PDvec; #vector 4096 by 1
targetContrast = PD; #64 by 64
epochs = 10;
#mainLoss = 100000;
def train(opt, input, targetContrast):
    with tf.GradientTape(persistent=True) as tape:
        vars = [alpha, deltat, gradientX, gradientY]; #gradients, deltat];
        tape.watch(vars);
        #print(tape.watched_variables());
        #groundTruthSigMatrix = tf.ones([8, 8], dtype=tf.complex64);
        result = forward(input, T1, T2, alpha, gradientX, gradientY, deltat, Nrep, Nactions, xVec, yVec);
        rec = tf.math.abs(reconstruction(result));
        loss = cost(rec, targetContrast);
        #loss_fn = cost(result, input);
        #phase = tf.constant(m.pi, dtype=tf.complex64)
        #groundTruthSignal = tf.constant([1.5], dtype=tf.complex64);
        #ez = tf.constant([0, 0, 1], dtype=tf.complex64, shape=(1, 3))
        #inputM = tf.matmul(input, ez);
        #result = signal(inputM, gradients[1,1], deltat[1,1], phase, gammaH, xVec, yVec);
        grads = tape.gradient(loss, vars)
        opt.apply_gradients(zip(grads, vars))
    return rec, loss;

#Change to make it in loss dependent
l = [];
for i in range(0, epochs):
    resArr = train(opt, input, targetContrast);
    plt.figure(1);
    a = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(targetContrast)), [Nrep ,Nactions]));
    plt.figure(2);
    b = plt.imshow(tf.reshape(tf.math.abs(tf.math.real(resArr[0])), [Nrep, Nactions]));
    plt.gray()
    plt.show()
    l.append(resArr[1]);
x = np.linspace(0, epochs, epochs+1)
plt.plot(x, l)
print(alpha)
print(deltat)
print(gradients)