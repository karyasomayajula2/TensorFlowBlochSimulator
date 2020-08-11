import tensorflow as tf
import scipy as sp
import numpy as np
import data as d

#Preprocess Data
inputData = d.data.imgCreator(0, 8, 8, 10, 25, 1);
Nrep = inputData[0].shape[0];
Nactions = inputData[0].shape[1];
PDArray = inputData[0].flatten();
PD = tf.constant(PDArray, dtype=tf.float64, shape=(64, 1));
T1 = inputData[1];
T2 = inputData[2];
xVec = tf.cast(tf.linspace(0, 8, 9), tf.complex64);
yVec = tf.cast(tf.linspace(0, 8, 9), tf.complex64);
alpha = tf.Variable(tf.ones([Nrep, Nactions], tf.double));
deltat = tf.Variable(tf.ones([Nrep, Nactions], tf.double));
gradients = tf.Variable(tf.ones([Nrep, Nactions], tf.double));

#Bloch Simulator Functions
gammaH = 42.575 * (2 * np.pi)

def xrot(phi):
    RotX = tf.constant([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]], dtype=tf.float64)
    return RotX;

def yrot(phi):
    RotY = tf.constant([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]], dtype=tf.float64)
    return RotY;

def zrot(phi):
    Rotz = tf.constant([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]], dtype=tf.float64)
    return Rotz;

def throt(phi, phase):
    Rz = zrot(-phase);
    Rx = xrot(phi);
    Rzinv = tf.linalg.inv(Rz);
    Rzinvx = tf.matmul(Rzinv, Rx);
    Rthrot = tf.matmul(Rzinvx, Rz);
    return Rthrot;

def relax(deltat, T1, T2):
    rel = tf.constant([[np.exp(-deltat / T2), 0, 0], [0, np.exp(-deltat / T2), 0], [0, 0, np.exp(-deltat / T1)]], dtype=tf.float64);
    return rel;

def freeprecess(deltat, df):  # for us relax does what freeprecess should do except the zrotation
    phi = 2 * (np.pi) * df * deltat / 1000;
    # E1 = np.exp(-deltat / T1);
    # E2 = np.exp(-deltat / T2);
    b = zrot(phi);
    # Afp = tf.matmul(tf.constant([[E2, 0, 0], [0, E2, 0], [0, 0, E1]], dtype= tf.float64), b);
    # Bfp = tf.constant([[0], [0], [1 - E1]], dtype=tf.float64)
    return b;

def gradprecess(m, gradient, deltat, rfPhase, gammaH, x, y):
    gx = gradient * np.math.cos(rfPhase);
    gy = gradient * np.math.sin(rfPhase);
    comp = tf.constant([0 - 1j*gammaH], dtype=tf.complex64); # -1i
    func = gx*x*deltat + gy*y*deltat;
    #exponential = tf.constant([func], dtype=tf.complex64, shape=(1,1));
    z = tf.multiply(comp, func);
    precess = tf.constant([np.exp(z.numpy()[0])], dtype=tf.complex64)
    ez = tf.constant([[0], [0], [1]], dtype = tf.float64)
    mz = tf.matmul(m, ez)
    mz = tf.cast(mz, tf.complex64);
    msig = tf.matmul(mz, precess); #check matrix dimensions for this portion
    return msig;

def signal(m, gradients, deltat, rfPhase, gammaH, x, y):
    #x[tf.newaxis, :], y[: , tf.newaxis]]
    svec = tf.reduce_sum(gradprecess(m, gradients, deltat, rfPhase, gammaH, x[tf.newaxis, :], y[: , tf.newaxis]));
    s = tf.reduce_sum(svec);
    return s;

def forward(PD, T1, T2, alpha, gradients, deltat, x, y, xVec, yVec, rfPhase=np.pi): #Pd is a tensor, T1 and T2 are scalars for that image
    ez = tf.constant([0, 0, 1], dtype=tf.float64, shape=(1, 3))
    s = np.zeros([Nrep, Nactions], dtype=complex)
    m0 = tf.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r,a].numpy(), rfPhase);
            m = tf.matmul(m, flip) # 64by3 times 3by3 = 64by3
            print(m)
            rel = relax(deltat[r,a].numpy(), T1, T2);
            m = tf.add(tf.matmul(m, rel), tf.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r,a].numpy(), df=10);
            m = tf.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            #ms = gradprecess(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, x, y) #64 by 1 vector transverse magnetiztion
            s[r,a] = signal(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, xVec, yVec);
    return s;



s = forward(PD, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec);
print(s);

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001);


#def train(optimizer, input):
 #   with tf.GradientTape() as tape:

