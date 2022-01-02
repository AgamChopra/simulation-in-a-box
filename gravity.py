import sys
sys.path.append('E:\evolution')
import torch
from cells import *

G = 6.67408E-11 # [m^3,kg^-1,s^-2]

class cell_physics():
    
    def __init__(self,mass,pos=[0,0],dx=0,dy=0): #vx=dx since we assume each step = 1 sec.
    
        self.dx=dx
        self.dy=dy
        self.mass = mass
        self.pos = pos
        
    def update(self,xmax,ymax,global_info):
        
        d = global_info[0]
        m2 = global_info[1]
        dX = d - torch.tensor((self.pos[0],self.pos[1])).unsqueeze(0)
        R2 = torch.sum(dX**2,dim=1).unsqueeze(1)
        F = G*self.mass*m2/R2
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
K=50
screen,_ = setup_environment(900,1600)
o = []
op = []
[op.append(cell_physics(random.randint(1E9,1E10),[random.randint(0,1600),random.randint(0,900)],random.randint(-10,10)/10,random.randint(-10,10)/10)) for _ in range(K)]
op.append(cell_physics(1E12,[1600/2,900/2],0,0))
[o.append(cell(screen,Point(op[i].pos[0],op[i].pos[1]),[random.randint(200, 250), random.randint(200, 250), random.randint(200, 250)],3)) for i in range(K)]
o.append(cell(screen,Point(op[-1].pos[0],op[-1].pos[1]),[200,0,0],6))
screen.getKey()
k=0.1
m2 = [op[i].mass for i in range(K+1)]
m2 = torch.tensor((m2)).unsqueeze(1)
while True:
   d = [[op[i].pos[0],op[i].pos[1]] for i in range(K+1)]
   d = torch.tensor((d))
   for i in range(K+1):
       mask = []
       for j in range(K+1):
           if j!= i:
               mask.append(j)
       #print(d[mask].shape,m2[mask].shape)
       global_info = [d[mask],m2[mask]]
       op[i].update(1600,900,global_info)
       o[i].update(float(op[i].dx), float(op[i].dy))
   time.sleep(k)   
#%%
a=torch.tensor(((2,1),(2,1),(1,2),(1,1),(4,3)))
m=torch.tensor((1,1)).unsqueeze(0)
print(a-m)