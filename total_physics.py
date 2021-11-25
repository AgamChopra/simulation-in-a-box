from cells import *
import torch
import sys
sys.path.append('E:\evolution')

C = 8.98755E9
G = 6.67408E-11
N = 1E-31
N2 = 1E-31
epsilon = 1E-9
g = 0  # 9.8#grav*
lam = 0  # -0.05#Friction*


class cell_physics():

    # vx=dx since we assume each step = 1 sec.
    def __init__(self, charge, mass, pos=[0, 0], dx=0, dy=0):

        self.dx = dx
        self.dy = dy
        self.charge = charge
        self.mass = mass
        self.pos = pos

    def update(self, xmax, ymax, global_info):

        d = global_info[0]
        m2 = global_info[1]
        q2 = global_info[2]
        dX = d - torch.tensor((self.pos[0], self.pos[1])).unsqueeze(0).float()
        R2 = torch.sum(dX**2, dim=1).unsqueeze(1) + epsilon
        R4 = torch.sum(dX**4, dim=1).unsqueeze(1) + epsilon
        F = ((G*self.mass*m2 - C*self.charge*q2)/R2) - (N/R4) + (N2*torch.where(q2 == self.charge, 1, 0) *
                                                                 torch.where(m2 == self.mass, 1, 0)/R2)  # <- attracting like particles to each other using kronecker delta
        dr = torch.sum((F*dX/(R2**(1/2)))/self.mass, dim=0) + \
            torch.tensor((lam*self.dx, lam*self.dy + g))
        self.dx += dr[0]
        self.dy += dr[1]
        if self.pos[0] > xmax or self.pos[0] < 10:
            self.dx = -self.dx  # /1.01
        if self.pos[1] > ymax or self.pos[1] < 10:
            self.dy = -self.dy  # /1.01
        self.pos[0] += self.dx
        self.pos[1] += self.dy
        return self.dx, self.dy


# %%
K = 100
screen, _ = setup_environment(900, 1600)
o = []
op = []
charge = [-1.6E-19, 1.6E-19]  # [1E-9,1E-9]#1.67E-9
[op.append(cell_physics(charge[1], 1.67E-27, [random.randint(-i*2+800, i*2+800), random.randint(-i*2+450, i*2+450)], 0, 0))
 for i in range(25)]  # [random.randint(0,1600),random.randint(0,900)]
[o.append(cell(screen, Point(op[i].pos[0], op[i].pos[1]), [random.randint(
    200, 250), random.randint(70, 100), random.randint(70, 100)], 2)) for i in range(25)]
[op.append(cell_physics(charge[0], 1E-30, [random.randint(-i*2+800, i*2+800), random.randint(-i*2+450, i*2+450)], 0, 0))
 for i in range(25, 50)]  # [random.randint(0,1600),random.randint(0,900)]
[o.append(cell(screen, Point(op[i].pos[0], op[i].pos[1]), [random.randint(
    70, 100), random.randint(70, 100), random.randint(200, 250)], 2)) for i in range(25, 50)]
[op.append(cell_physics(0, 1E-25, [random.randint(-i+800, i+800), random.randint(-i+450, i+450)], 0, 0))
 for i in range(50, 100)]  # [random.randint(0,1600),random.randint(0,900)]
[o.append(cell(screen, Point(op[i].pos[0], op[i].pos[1]), [random.randint(70, 100),
          random.randint(200, 250), random.randint(70, 100)], 2)) for i in range(50, 100)]
screen.getKey()
k = 0.10
q2 = [op[i].charge for i in range(K)]
q2 = torch.tensor((q2)).unsqueeze(1)
m2 = [op[i].mass for i in range(K)]
m2 = torch.tensor((m2)).unsqueeze(1)
while True:
    d = [[op[i].pos[0], op[i].pos[1]] for i in range(K)]
    d = torch.tensor((d))
    for i in range(K):
        mask = []
        for j in range(K):
            if j != i:
                mask.append(j)
        global_info = [d[mask], m2[mask], q2[mask]]
        op[i].update(1600-10, 900-10, global_info)
        o[i].update(float(op[i].dx), float(op[i].dy))
    time.sleep(k)
# %%
'''
q2 = torch.tensor(((1),(3),(2E-19),(2E-19))).unsqueeze(1).float()
m2 = torch.tensor(((1),(2),(3E-19),(1E-19))).unsqueeze(1).float()
R2 = torch.tensor(((1,1),(2,1),(1,3),(2,3))).float()
charge = float(2E-19)
mass = float(3E-19)
print(N2*torch.where(q2==charge,1,0)*torch.where(m2==mass,1,0)/R2)#*R4
'''
