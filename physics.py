import torch
from matplotlib import pyplot as plt
import random
import numpy

# Medium = H2O. Physical properties of H2O -> https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm

temp = []
mu = []
solar = []
n1 = []
n2 = []
T_coeff, T_bar, a, b = 21, torch.tensor((1.5,4.5,15)), torch.tensor((-0.2,-0.005, -0.00054)), torch.tensor((0,0,0))
S_bar, a_, b_ = torch.tensor((20,25)), torch.tensor((-0.001,-0.0008)), torch.tensor((0,0))
N1_bar, a_N1, b_N1 = torch.tensor((2,5)), torch.tensor((-0.002,-0.00054)), torch.tensor((0,0))
N2_bar, a_N2, b_N2 = torch.tensor((4,1)), torch.tensor((-0.008,-0.0001)), torch.tensor((0,0))

def T(t, T_coeff, T_bar, a, b, Solar):
    return T_coeff + torch.sum(T_bar * torch.sin(a*t + b)) + Solar/torch.pi

def MU(T):
    # coefficient of viscosity for H2O. Function of Temp. [Kg s^-1 m-1]
    return 3E-08 * T ** 4 - 9E-06 * T ** 3 + 0.001 * T ** 2 - 0.0552 * T + 1.7784

def Fd(T,r,v):
    '''
    # Drag Force (Stokes’ Law) Fd = - 6 π η r v
    # Example:
        T = 33
        r, v = torch.ones((1600,900,1)), torch.ones((1600,900,2))
        print(Fd(T,r,v).shape)
    '''
    return - 6 * torch.pi * MU(T) * (r * v)

def Solar_E(t, bar, a, b):
    # Solar Energy. Amount of Energy available per photo_org_neuron per timestep.
    return torch.log(20 + torch.abs(torch.sum(bar * torch.sin(a*t + b)) + torch.exp(torch.pi * torch.rand(())))).to(dtype=torch.int32)

def N1(t, bar, a, b):
    return 2 + torch.abs(torch.sum(bar * torch.sin(a*t + b)) + torch.exp(torch.rand(())))

def N2(t, bar, a, b):
    return 2 + torch.abs(torch.sum(bar * torch.sin(a*t + b)) + torch.exp(torch.rand(())))

def main():
    for i in range(0,100):
        solar.append(Solar_E(i, S_bar, a_, b_))
        temp.append(T(i,T_coeff, T_bar, a, b,solar[-1]))
        mu.append(MU(temp[-1]))
        n1.append(N1(i,N1_bar, a_N1, b_N1))
        n2.append(N2(i,N2_bar, a_N2, b_N2))
           
    print(min(temp),max(temp))
    print(min(mu),max(mu))
    print(min(solar),max(solar))
    print(min(n1),max(n1))
    print(min(n2),max(n2))
     
    plt.plot(temp,'black') 
    plt.title('Temp')
    plt.show()  
    plt.plot(mu,'r')
    plt.title('Visc. of H2O medium')
    plt.show()
    plt.plot(solar,'orange')
    plt.title('Solar Energy')
    plt.show()
    plt.plot(n1,'blue')
    plt.title('N1')
    plt.show()
    plt.plot(n2,'green')
    plt.title('N2')
    plt.show()
#%%   
if __name__ == '__main__':
    main()
#%%

cell_dynamics = torch.randint(90,(400, 4)).to(dtype=torch.float)#[Cells[Alive],4]  x, y, vx, vy
color = torch.randint(255, (400, 3))

#print(cell_dynamics)
#print(color)

#Only Collision Physics:
    #Rules:
        #If Particles have a distance <= COLL_DIST, they collide and stop. We assume that the cells are very squishy and sticky in a relatively dense medium.
        
COLL_DIST = 1.

v = cell_dynamics[:,2:] / 10
#print('velo:',v)

a = cell_dynamics[:,:2]
#print('pos:',a)

dist = torch.cdist(a, a)
#print('dist:',dist)

v_col = torch.where(dist > COLL_DIST, 1., 0.)
#print(v_col)

v_col = torch.sum(v_col,dim=1)
#print(v_col)

theta = v.shape[0] - 1
#print(theta)

v_col = torch.where(v_col < theta, 0., 1.)
#print(v_col)

v = v * v_col.view(v.shape[0],1)
#print(v)

#pos update -> arty -> rend
SPF = 1 # 1 sec/frame

dx = v * SPF
#print(dx)

pos = (a + dx).to(dtype = torch.int32)
#print(pos)

arty = torch.cat((pos,color),dim = 1)
#print(arty)