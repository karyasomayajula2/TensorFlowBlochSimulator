import scipy as sp
import numpy as np
import data as d
import torch
import matplotlib.pyplot as plt
import math
# Preprocess Data
inputData = d.data.getSheppLogan(0);#d.data.imgCreator(0, 8, 8, 10, 25, 1);
Nrep = 32 #inputData[0].shape[0];
Nactions = 32 #inputData[0].shape[1];
PDArray = inputData#[0].flatten();
PD = torch.as_tensor(PDArray, dtype=torch.float64);
PD = torch.reshape(PD,(Nrep,Nactions))
PDvec = torch.reshape(PD, [Nrep*Nactions, 1])
T1 = 2.0 #inputData[1];
T2 = 20.0 #inputData[2];
xVec = torch.linspace(1, 32, 32);
yVec = torch.linspace(1, 32, 32);
xVec.type(torch.complex64);
yVec.type(torch.complex64);
alpha = torch.zeros(Nrep, Nactions, requires_grad=True)  # , torch.double));
deltat = torch.ones(Nrep, Nactions, requires_grad=True)  # , torch.double));
gx = torch.linspace(0, 10, Nrep*Nactions) #torch.zeros(Nrep, Nactions, requires_grad=True)
gy = torch.linspace(0, 10, Nrep*Nactions)#torch.zeros(Nrep, Nactions, requires_grad=True)  # , torch.double));
gx = torch.reshape(gx, (Nrep, Nactions));
gy = torch.reshape(gy, (Nrep, Nactions));

# Bloch Simulator Functions
gammaH = 42.575*(2 * np.pi)


def xrot(phi):
    # RotX = torch.float64(torch.tensor([torch.tensor([1, 0, 0]), [0, torch.cos(phi), -torch.sin(phi)], [0, torch.sin(phi), torch.cos(phi)]]))
    RotX = torch.tensor([[torch.cos(phi), -torch.sin(phi), 0],
                         [torch.sin(phi), torch.cos(phi), 0], [0, 0, 1]],
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
#    phi = torch.tensor(phi)
#    phase = torch.tensor(phase)
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


def freeprecess(deltat):  # for us relax does what freeprecess should do except the zrotation
    phi = deltat*gammaH*1.5; #3 is also a common value units for B0 are tesla
    # E1 = np.exp(-deltat / T1);
    # E2 = np.exp(-deltat / T2);
    b = zrot(phi);
    # Afp = tf.matmul(tf.constant([[E2, 0, 0], [0, E2, 0], [0, 0, E1]], dtype= tf.float64), b);
    # Bfp = tf.constant([[0], [0], [1 - E1]], dtype=tf.float64)
    return b;

def gradprecess(m, gx, gy, deltat, rfPhase, x, y):
    #gx = gradient * torch.cos(rfPhase);
    #gy = gradient * torch.sin(rfPhase);
    comp = torch.tensor([0 + 1j], dtype=torch.complex64);  # -1i
    #func = gx * x * deltat + gy * y * deltat; #this is not necessary since we aren't using gradient moments, rather we are using an overall gradient input
    func = torch.add(gx * x, gy * y)
    #    exponential = torch.tensor([func], dtype=torch.complex64);
    #    exponential = torch.reshape(exponential, shape=(1, 1))
    mx = m[:, 0]
    my = m[:, 1]
    mxy = torch.sqrt(torch.square(mx)+torch.square(my));
    temp = torch.mul(comp, func);
    precess = torch.exp(temp);
    precessLine = torch.reshape(precess, (Nrep*Nactions, 1))
    # precess = torch.complex64(precess);
    # torch.tensor(torch.exp(torch.index_select(z, 0, 0))#, dtype=torch.complex64)
    #ez = torch.tensor([[0], [0], [1]], dtype=torch.float64);
    #mz = torch.matmul(m, ez);
    #mz = mz.type(torch.complex64);
    msig = torch.mul(mxy, precessLine)
    # msig = torch.mm(mz, precess);  # check matrix dimensions for this portion L1 norm
    # mmcombined = mz.real + mz.imag;
    # precessLinec = precessLine.real + precessLine.imag;
    # msig = torch.mul(mmcombined, precessLinec)
    return msig;


def signal(m, gx, gy, deltat, rfPhase, x, y):
    # svec = torch.sum(m((x, y)));
    x = torch.reshape(x, [1, 32]);
    y = torch.reshape(y, [32, 1]);
    svec = torch.sum(gradprecess(m, gx, gy, deltat, rfPhase, x, y));
    return svec;


def forward(PD, T1, T2, alpha, gx, gy, deltat, x, y, xVec, yVec, rfPhase=0.0):
            #rfPhase=torch.tensor(math.pi)):  # Pd is a tensor,PD T1 and T2 are scalars for that image
    ez = torch.tensor([0, 0, 1], dtype=torch.float64)
    ez = torch.reshape(ez, (1, 3))
    # print(torch.is_tensor(ez))
    s = torch.zeros(Nrep, Nactions, dtype=torch.complex64)  # dtype=complex)
    m0 = PD*ez  # torch.matmul(PD, ez); #if 8by8 image PD is a 64by1 matrix and m is now a 64by3: 64by1*1by3 = 64by3
    m = m0;
    phase = torch.tensor(rfPhase, dtype=torch.float64)
    t1 = torch.tensor(T1, dtype=torch.complex64)
    t2 = torch.tensor(T2, dtype=torch.complex64)
    for r in range(0, Nrep):
        for a in range(0, Nactions):
            flip = throt(alpha[r, a], phase);
            m = torch.mm(m, flip)  # 64by3 times 3by3 = 64by3
            rel = relax(deltat[r, a], T1, T2);
            m = torch.add(torch.matmul(m, rel),
                          torch.matmul(m0, 1 - rel));  # 64by3 times 3by3 + 64by3 original time 3by3 = 64by3
            b = freeprecess(deltat[r, a])#,#);
            m = torch.matmul(m, b);  # 64by3 times 3by3 = zrotated mag vectors
            # ms = gradprecess(m, gradients[r, a], deltat[r, a], rfPhase, gammaH, x,
            # y)  # 64 by 1 vector transverse magnetiztion
            # s[r, a] = signal(ms, xVec, yVec);
            kx = gx[r, a].item();
            kx = int(kx)
#            kx = kx.data()[0]
            ky = gy[r, a].item();
            ky = int(ky)
#            ky = ky.data()[0]
            s[kx, ky] = signal(m, gx[r, a], gy[r, a], deltat[r, a], rfPhase, xVec, yVec);
    return s;


def reconstruction(s):
    ss = torch.stack([s.real, s.imag], 2)
    rec = torch.ifft(ss, 2)
    recon = rec[:, :, 1]
    recon = torch.reshape(recon, [Nrep*Nactions, 1])
    #recon = recon.sum(1)
    # recon = torch.reshape(recon,[8,8]) reshape for if you want like an image, if not just leave as 64 x 1 for loss calc
    # reci = torch.imag(recon[:, 1])
    # recr = torch.tensor(recon[:, 0], dtype=torch.complex64)
    # recon = recr + reci;
    return recon


def cost(recon, PD):
    loss = torch.square(PD - recon);
    cost = torch.view_as_complex(torch.mean(loss));
    return cost


# vars = [alpha, deltat, gx, gy]
# opt = torch.optim.Adam(vars, lr=.001)
# input = PD;
# epochs = 1000;
# mainLoss = 100000;


def train(opt, input):
    # torch.autograd:
    #vars.requires_grad_(True)
    result = forward(input, T1, T2, alpha, gx, gy, deltat, Nrep, Nactions, xVec, yVec);
    rec = reconstruction(result);
    loss = cost(rec, input);
    mainloss = loss
    grads = loss.backward(loss, retain_graph=True)
    vars.optimizer.step()
    return mainloss


# # Change to make it in loss dependent
# print(mainLoss);
# while mainLoss > 0.1:
#     train(opt, input);
# print(alpha)
# print(deltat)
# print(gx)
# print(gy)
# msig = gradprecess([0,0,1],10,deltat,torch.tensor(10),torch.tensor(10),1,1)
# print(msig)
s = forward(PDvec, T1, T2, alpha, gx, gy, deltat, Nrep, Nactions, xVec, yVec);
#print(s);
recon = reconstruction(s)
#print(recon)
recons = torch.reshape(recon, [Nrep, Nactions])
plt.figure()
plt.imshow(recons.detach().numpy())
plt.show()
#plt.figure1
# recon.detach()
# plt.imshow(torch.reshape(recon, [Nrep, Nactions]))
