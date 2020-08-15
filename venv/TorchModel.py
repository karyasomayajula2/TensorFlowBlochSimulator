import scipy as sp
import numpy as np
import data as d
import torch

#Preprocess Data
inputData = d.data.imgCreator(0, 8, 8, 10, 25, 1);
Nrep = inputData[0].shape[0];
Nactions = inputData[0].shape[1];
PDArray = inputData[0].flatten();
PD = torch.as_tensor(PDArray, dtype=torch.float64);
PD = torch.reshape(PD,shape=(64, 1));
T1 = inputData[1];
T2 = inputData[2];
xVec = torch.linspace(0, 8, 9);
yVec = torch.linspace(0, 8, 9);
alpha = torch.autograd.Variable(torch.zeros([Nrep, Nactions]))#, torch.double));
deltat = torch.autograd.Variable(torch.zeros([Nrep, Nactions]))#, torch.double));
gradients =torch.autograd.Variable(torch.zeros([Nrep, Nactions]))#, torch.double));

#Bloch Simulator Functions
gammaH = 42.575 * (2 * np.pi)

def xrot(phi):
    RotX = torch.as_tensor([[1, 0, 0], [0, torch.cos(phi), -torch.sin(phi)], [0, torch.sin(phi), torch.cos(phi)]], dtype=torch.float64)
    return RotX;

def yrot(phi):
    RotY = torch.as_tensor([[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]], dtype=torch.float64)
    return RotY;

def zrot(phi):
    Rotz = torch.as_tensor([[torch.cos(phi), -torch.sin(phi), 0], [torch.sin(phi), torch.cos(phi), 0], [0, 0, 1]], dtype=torch.float64)
    return Rotz;

def throt(phi, phase):
    Rz = zrot(-phase);
    Rx = xrot(phi);
    Rzinv = torch.inverse(Rz);
    Rzinvx = torch.matmul(Rzinv, Rx);
    Rthrot = torch.matmul(Rzinvx, Rz);
    return Rthrot;

def relax(deltat, T1, T2):
    rel = torch.as_tensor([[torch.exp(-deltat / T2), 0, 0], [0, torch.exp(-deltat / T2), 0], [0, 0, torch.exp(-deltat / T1)]], dtype=torch.float64);
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
    gx = gradient * torch.cos(rfPhase);
    gy = gradient * torch.sin(rfPhase);
    comp = torch.as_tensor([0 - 1j*gammaH], dtype=torch.complex64); # -1i
    func = gx*x*deltat + gy*y*deltat;
    exponential = torch.as_tensor([func], dtype=torch.complex64);
    exponential = torch.reshape(exponential,shape=(1,1))
    z = torch.mul(comp, exponential);
    precess = torch.as_tensor([torch.exp(z.numpy()[0])], dtype=torch.complex64)
    ez = torch.as_tensor([[0], [0], [1]], dtype = torch.float64)
    mz = torch.matmul(m, ez)
    mz = torch.complex64(mz)
    msig = torch.matmul(mz, precess); #check matrix dimensions for this portion
    return msig;

def signal(m, x, y):
    svec = torch.sum(m((x[np.newaxis, :], y[: , np.newaxis])));
    s = torch.sum(svec);
    return s;

def forward(PD, T1, T2, alpha, gradients, deltat, x, y, xVec, yVec, rfPhase=np.pi): #Pd is a tensor, T1 and T2 are scalars for that image
    ez = torch.tensor([0,0,1], dtype=torch.float64)
    #print(torch.is_tensor(ez))
    s = np.zeros([Nrep, Nactions], dtype=complex)
    m0 = torch.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r,a].numpy(), rfPhase);
            m = torch.mm(m, flip) # 64by3 times 3by3 = 64by3
            rel = relax(deltat[r,a].numpy(), T1, T2);
            m = torch.add(torch.matmul(m, rel), torch.matmul(m0, 1-rel)); # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r,a].numpy(), df=10);
            m = torch.matmul(m, b); #64by3 times 3by3 = zrotated mag vectors
            ms = gradprecess(m, gradients[r,a].numpy(), deltat[r,a].numpy(), rfPhase, gammaH, x, y) #64 by 1 vector transverse magnetiztion
            s[r,a] = signal(ms, xVec, yVec);
    return s;

#msig = gradprecess([0,0,1],10,deltat,torch.tensor(10),torch.tensor(10),1,1)
#print(msig)
s = forward(PD, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec);
print(s);