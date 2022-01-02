import sys
sys.path.append('E:\evolution')
import torch
from cells import *

C = 8.98755E9 # [m^3,kg^-1,s^-2]

class cell_physics():
    
    def __init__(self,charge,mass,pos=[0,0],dx=0,dy=0): #vx=dx since we assume each step = 1 sec.
    
        self.dx=dx
        self.dy=dy
        self.charge = charge
        self.mass = mass
        self.pos = pos
        
    def update(self,xmax,ymax,global_info):
        
        d = global_info[0]
        q2 = global_info[1]
        dX = d - torch.tensor((self.pos[0],self.pos[1])).unsqueeze(0)
        R2 = torch.sum(dX**2,dim=1).unsqueeze(1)
        F = -C*self.charge*q2/R2
        dr = torch.sum((F*dX/(R2**(1/2)))/self.mass,dim=0)
        self.dx += dr[0]
        self.dy += dr[1]
        if self.pos[0]>xmax or self.pos[0]< 0:
            self.dx = -self.dx/1.01
        if self.pos[1]>ymax or self.pos[1]< 0:
            self.dy = -self.dy/1.01
        self.pos[0] += self.dx
        self.pos[1] += self.dy
        return self.dx, self.dy
        
#%%
K=200
screen,_ = setup_environment(900,1600)
o = []
op = []
charge = [100,100]
[op.append(cell_physics(charge[random.randint(0,1)],random.randint(1,10)/1E-12,[random.randint(0,1600),random.randint(0,900)],0,0)) for _ in range(K)]
[o.append(cell(screen,Point(op[i].pos[0],op[i].pos[1]),[random.randint(200, 250), random.randint(200, 250), random.randint(200, 250)],3)) for i in range(K)]
screen.getKey()
k=0.1
q2 = [op[i].charge for i in range(K)]
q2 = torch.tensor((q2)).unsqueeze(1)
while True:
   d = [[op[i].pos[0],op[i].pos[1]] for i in range(K)]
   d = torch.tensor((d))
   for i in range(K):
       mask = []
       for j in range(K):
           if j!= i:
               mask.append(j)
       #print(d[mask].shape,m2[mask].shape)
       global_info = [d[mask],q2[mask]]
       op[i].update(1600,900,global_info)
       o[i].update(float(op[i].dx), float(op[i].dy))
   time.sleep(k)   
#%%
a=torch.tensor(((2,1),(2,1),(1,2),(1,1),(4,3)))
m=torch.tensor((1,1)).unsqueeze(0)
print(a-m)