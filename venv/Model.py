import tensorflow as tf
import scipy as sp
import numpy as np
import data as d
import math as m

#Preprocess Data
inputData = d.data.imgCreator(0, 8, 8, 10, 25, 1);
Nrep = inputData[0].shape[0];
Nactions = inputData[0].shape[1];
PDArray = inputData[0].flatten();
PD = tf.constant(PDArray, dtype=tf.float64, shape=(64, 1));
T1 = inputData[1];
T2 = inputData[2];
xVec = tf.cast(tf.linspace(1, 8, 8), tf.float64);
yVec = tf.cast(tf.linspace(1, 8, 8), tf.float64);
alpha = tf.Variable(tf.ones([Nrep, Nactions], tf.float64));
deltat = tf.Variable(tf.ones([Nrep, Nactions], tf.float64));
gradients = tf.Variable(tf.ones([Nrep, Nactions], tf.float64));

#Bloch Simulator Functions
gammaH = 42.575 * (2 * m.pi)

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

def freeprecess(deltat, phase, df):  # for us relax does what freeprecess should do except the zrotation
    phi = 2 * (phase) * df * deltat / 1000;
    p = tf.constant(phi, dtype = tf.float64);
    # E1 = np.exp(-deltat / T1);
    # E2 = np.exp(-deltat / T2);
    b = zrot(phi);
    # Afp = tf.matmul(tf.constant([[E2, 0, 0], [0, E2, 0], [0, 0, E1]], dtype= tf.float64), b);
    # Bfp = tf.constant([[0], [0], [1 - E1]], dtype=tf.float64)
    return b;

def gradprecess(m, gradient, deltat, phase, gammaH, x, y):
    gx = gradient * tf.cos(phase);
    gy = gradient * tf.sin(phase);
    comp = tf.constant([0 - 1j*gammaH], dtype=tf.complex64); # -1i
    func = tf.add(gx*x*deltat, gy*y*deltat);
    #exponential = tf.constant([func], dtype=tf.complex64, shape=(1,1));
    f = tf.cast(func, dtype=tf.complex64);
    z = tf.multiply(comp, f);
    precess = tf.exp(z);
    precessLine = tf.reshape(precess, [1,64])
    ez = tf.constant([[0], [0], [1]], dtype = tf.float64)
    mz = tf.matmul(m, ez)
    mz = tf.cast(mz, tf.complex64);
    msig = tf.matmul(mz, precessLine); #check matrix dimensions for this portion
    return msig;

def signal(m, gradients, deltat, rfPhase, gammaH, x, y):
    #x[tf.newaxis, :], y[: , tf.newaxis]]
    svec = tf.reduce_sum(gradprecess(m, gradients, deltat, rfPhase, gammaH, x[tf.newaxis, :], y[: , tf.newaxis]));
    s = tf.reduce_sum(svec);
    return s;

def forward(PD, T1, T2, alpha, gradients, deltat, x, y, xVec, yVec, rfPhase=m.pi): #Pd is a tensor, T1 and T2 are scalars for that image
    ez = tf.constant([0, 0, 1], dtype=tf.float64, shape=(1, 3))
    s = np.zeros([Nrep, Nactions], dtype=complex)
    m0 = tf.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    t1 = tf.constant(T1, dtype = tf.float64)
    t2 = tf.constant(T2, dtype = tf.float64)
    phase = tf.constant(rfPhase, dtype=tf.float64)
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r,a], phase); #no .numpy() anywhere
            m = tf.matmul(m, flip) # 64by3 times 3by3 = 64by3y
            rel = relax(deltat[r,a], t1, t2);
            m = tf.add(tf.matmul(m, rel), tf.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r,a], phase, df=10);
            m = tf.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            #ms = gradprecess(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, x, y) #64 by 1 vector transverse magnetiztion
            s[r,a] = signal(m, gradients[r,a], deltat[r,a], phase, gammaH, xVec, yVec);
    return s;



#s = forward(PD, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec);
#print(s);

def reconstruction(s):
    rec = tf.signal.ifft2d(s);
    recon = tf.math.real(rec)+tf.math.imag(rec); #use abs or real
    recon = tf.cast(recon, tf.float64);
    recon = tf.reshape(recon, [64,1]);
    return recon;


def cost(recon, PD):
    loss = tf.square(PD-recon);
    cost = tf.reduce_mean(loss);
    return cost;

opt = tf.keras.optimizers.Adam(learning_rate=0.001);
input = PD; #in the future used to make training batches
epochs = 10;
def train(opt, input):
    with tf.GradientTape(persistent=True) as tape:
        vars = [alpha, gradients, deltat];
        tape.watch(vars);
        print(tape.watched_variables());
        result = reconstruction(forward(input, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec))
        loss_fn = cost(result, input);
        grads = tape.gradient(loss_fn, vars)
        opt.apply_gradients(zip(grads, vars))

for i in range(0, epochs):
    train(opt, input);
