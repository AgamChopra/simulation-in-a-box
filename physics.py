import torch
from math import sin
from matplotlib import pyplot as plt

# Medium = H2O. Physical properties of H2O -> https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm

PI = torch.pi

temp = []
mu = []
solar = []
n1 = []
n2 = []
T_coeff, T_bar, a_T, b_T = 21, torch.tensor((1.5,4.5,15)), torch.tensor((-0.2,-0.005, -0.00054)), torch.tensor((0,0,0))
S_bar, a_S, b_S = torch.tensor((20,25)), torch.tensor((-0.001,-0.0008)), torch.tensor((0,0))
N1_bar, a_N1, b_N1 = torch.tensor((2,5)), torch.tensor((-0.002,-0.00054)), torch.tensor((0,0))
N2_bar, a_N2, b_N2 = torch.tensor((4,1)), torch.tensor((-0.008,-0.0001)), torch.tensor((0,0))
day_night = 0.85


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
    return - 6 * PI * MU(T) * (r * v)


def Solar_E(t, bar, a, b):
    # Solar Energy. Amount of Energy available per photo_org_neuron per timestep.
    return torch.log(20 + torch.abs(torch.sum(bar * torch.sin(a*t + b))))


def N1(t, bar, a, b):
    return 2 + torch.abs(torch.sum(bar * torch.sin(a*t + b)) + torch.exp(torch.rand(())))


def N2(t, bar, a, b):
    return 2 + torch.abs(torch.sum(bar * torch.sin(a*t + b)) + torch.exp(torch.rand(())))


def dynamics(ringo, color, time_step, RADIUS = 10, COLL_DIST = 10., MASS = 1E-6, SPF = 1/144, WIDTH = 1600, HEIGHT = 900):   
    '''
    Drag and Collision Physics:
        Rules:
            For drag, velocity has to be assumed to be very small.
            If Particles have a distance <= COLL_DIST, they collide and stop. We assume that the cells are very squishy and sticky in a relatively dense medium.
        Variables:
            cell_dynamics. [N,4] tensor. [:,:2] -> x,y positions of the particles, [:,2:] -> vx,vy velocities of those particles after neural network updates.
            color. [N,3] tensor storing R,G,B color values corrosponding to the N cells. Read directly from the genes.
            time_step. Current time step of the simulation.
            Radius. Radius of the particles.
            COLL_DIST. min. collision distance for the particles.
            MASS. Assumed mass of the particles.
            SPF. seconds traverced per simulation step/frame. smaller SPF = higher accuracy but may require longer runs.
        Output:
            arty. tensor of shape [N,5], [N,positions(2),color(3)]
            cell_dynamics. updated cell positions and velocities after the physics step.
    '''
    v = ringo[:,2:]
    x = ringo[:,:2]
    
    SOLAR = Solar_E(time_step, S_bar, a_S, b_S) * (sin(time_step * day_night) + 1) * 0.5
    TEMP = T(time_step, T_coeff, T_bar, a_T, b_T, SOLAR)
    N_1 = N1(time_step, N1_bar, a_N1, b_N1)
    N_2 = N2(time_step, N2_bar, a_N2, b_N2)
    
    # Drag on the particle slowing down current velocity.
    F_drag = Fd(TEMP,RADIUS * 1E-6,v)
    dv_drag = F_drag * SPF / MASS
    v = v + (dv_drag * torch.where(torch.abs(v) < torch.abs(dv_drag), 0, 1)) - ( 0.1 * v * torch.where(torch.abs(v) < torch.abs(dv_drag), 1, 0)) # Condition to prevent incorrect drag force(to preserve Conservation of Energy)  when slow particle velocity assumption is broken. We assume in such a case that the fluid exerts a linear drag if velocity increases corrosponding to the particles min. velocity threshold in the medium.
    
    # Calculating collisions
    dist = torch.cdist(x, x)
    
    v_col = torch.where(dist > COLL_DIST, 1., 0.)
    v_col = torch.sum(v_col,dim=1)
    theta = v.shape[0] - 1
    v_col = torch.where(v_col < theta, 0., 1.)
    v = v * v_col.view(v.shape[0],1)
    dx = v * SPF
    x = x + dx
    
    # Applying boundry conditions 
    v = v * torch.cat((torch.where(((x[:,0] > WIDTH) * v[:,0]) > 0, -1, 1).reshape(x.shape[0], 1),
                        torch.where(((x[:,1] > HEIGHT) * v[:,1]) > 0, -1, 1).reshape(x.shape[0], 1)),1) *\
            torch.cat((torch.where(((x[:,0] < 0) * v[:,0]) < 0, -1, 1).reshape(x.shape[0], 1),
                        torch.where(((x[:,1] < 0) * v[:,1]) < 0, -1, 1).reshape(x.shape[0], 1)),1)
    x = torch.nan_to_num(x * torch.cat((torch.where(x[:,0] > WIDTH, torch.nan, 1.).reshape(x.shape[0], 1), torch.ones((x.shape[0],1))),1), nan=WIDTH)
    x = torch.nan_to_num(x * torch.cat((torch.ones((x.shape[0],1)),torch.where(x[:,1] > HEIGHT, torch.nan, 1.).reshape(x.shape[0], 1)),1), nan=HEIGHT)
    
    # Outputs
    arty = torch.cat((x.to(dtype = torch.int32),color),dim = 1)
    ringo = torch.cat((x,v),dim = 1)
    
    return arty, ringo, SOLAR, TEMP, N_1, N_2

def main():
    for i in range(0,100000):
        solar.append(Solar_E(i, S_bar, a_S, b_S))
        temp.append(T(i,T_coeff, T_bar, a_T, b_T,solar[-1]))
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
    
    cell_dynamics = torch.randint(30,(400, 4)).to(dtype=torch.float)#[Cells[Alive],4]  x, y, vx, vy
    color = torch.randint(255, (400, 3))
    print(dynamics(cell_dynamics,color,100))
    
#%%   
if __name__ == '__main__':
    main()
#%%