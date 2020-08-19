import scipy as sp
import numpy as np
import data as d
import torch

# Preprocess Data
inputData = d.data.imgCreator(0, 8, 8, 10, 25, 1);
Nrep = inputData[0].shape[0];
Nactions = inputData[0].shape[1];
PDArray = inputData[0].flatten();
PD = torch.as_tensor(PDArray, dtype=torch.float64);
PD = torch.reshape(PD, shape=(64, 1));
T1 = inputData[1];
T2 = inputData[2];
xVec = torch.linspace(0, 8, 9);
yVec = torch.linspace(0, 8, 9);
alpha = torch.autograd.Variable(torch.zeros(Nrep, Nactions))  # , torch.double));
deltat = torch.autograd.Variable(torch.zeros(Nrep, Nactions))  # , torch.double));
gradients = torch.autograd.Variable(torch.zeros(Nrep, Nactions))  # , torch.double));

# Bloch Simulator Functions
gammaH = 42.575 * (2 * np.pi)


def xrot(phi):
    # RotX = torch.float64(torch.tensor([torch.tensor([1, 0, 0]), [0, torch.cos(phi), -torch.sin(phi)], [0, torch.sin(phi), torch.cos(phi)]]))
    RotX = torch.tensor([[torch.cos(torch.tensor(phi)), -torch.sin(torch.tensor(phi)), 0],
                         [torch.sin(torch.tensor(phi)), torch.cos(torch.tensor(phi)), 0], [0, 0, 1]],
                        dtype=torch.float64)
    return RotX;


def yrot(phi):
    RotY = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]],
                        dtype=torch.float64)
    return RotY;


def zrot(phi):
    Rotz = torch.tensor([[torch.cos(phi), -torch.sin(phi), 0], [torch.sin(phi), torch.cos(phi), 0], [0, 0, 1]],
                        dtype=torch.float64)
    return Rotz;


def throt(phi, phase):
    phi = torch.tensor(phi)
    phase = torch.tensor(phase)
    Rz = zrot(-phase);
    Rx = xrot(phi);
    Rzinv = torch.inverse(Rz);
    Rzinvx = torch.matmul(Rzinv, Rx);
    Rthrot = torch.matmul(Rzinvx, Rz);
    return Rthrot;


def relax(deltat, T1, T2):
    rel = torch.tensor(
        [[torch.exp(-deltat / T2), 0, 0], [0, torch.exp(-deltat / T2), 0], [0, 0, torch.exp(-deltat / T1)]],
        dtype=torch.float64);
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
    gradient = torch.tensor(gradient)
    gx = gradient * torch.cos(rfPhase);
    gy = gradient * torch.sin(rfPhase);
    comp = torch.as_tensor([0 - 1j * gammaH], dtype=torch.complex64);  # -1i
    func = gx * x * deltat + gy * y * deltat;
#    exponential = torch.tensor([func], dtype=torch.complex64);
#    exponential = torch.reshape(exponential, shape=(1, 1))
    z = torch.mul(comp, func);
    precess = torch.exp(z);
    #precess = torch.complex64(precess);
    #torch.tensor(torch.exp(torch.index_select(z, 0, 0))#, dtype=torch.complex64)
    ez = torch.tensor([[0], [0], [1]], dtype=torch.float64);
    mz = torch.matmul(m, ez);
    mz = mz.type(torch.complex64);
    msig = torch.mm(mz, precess);  # check matrix dimensions for this portion
    return msig;


def signal(m, gradients, deltat, rfPhase, gammaH, x, y):
    x = x.reshape(1, 9)
    y = y.reshape(9, 1)
    #svec = torch.sum(m((x, y)));
    svec = torch.sum(gradprecess(m, gradients, deltat, rfPhase, gammaH, x, y));
    s = torch.sum(svec);
    return s;


def forward(PD, T1, T2, alpha, gradients, deltat, x, y, xVec, yVec,
            rfPhase=torch.tensor(np.pi)):  # Pd is a tensor, T1 and T2 are scalars for that image
    ez = torch.tensor([0, 0, 1], dtype=torch.float64)
    # print(torch.is_tensor(ez))
    s = torch.zeros(Nrep, Nactions)  # dtype=complex)
    m0 = PD * ez  # torch.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r, a], rfPhase);
            m = torch.mm(m, flip)  # 64by3 times 3by3 = 64by3
            rel = relax(torch.tensor(deltat[r, a]), T1, T2);
            m = torch.add(torch.matmul(m, rel),
                          torch.matmul(m0, 1 - rel));  # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(torch.tensor(deltat[r, a]), df=10);
            m = torch.matmul(m, b);  # 64by3 times 3by3 = zrotated mag vectors
            #ms = gradprecess(m, gradients[r, a], deltat[r, a], rfPhase, gammaH, x,
                             #y)  # 64 by 1 vector transverse magnetiztion
            #s[r, a] = signal(ms, xVec, yVec);
            s[r, a] = signal(m, gradients[r, a].numpy(), deltat[r, a], rfPhase, gammaH, xVec, yVec);
    return s;


# msig = gradprecess([0,0,1],10,deltat,torch.tensor(10),torch.tensor(10),1,1)
# print(msig)
s = forward(PD, T1, T2, alpha, gradients, deltat, Nrep, Nactions, xVec, yVec);
print(s);
