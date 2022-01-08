import pygame
import random
import torch

C = 8.98755E9
G = 6.67408E-11
N = 1E-14
N2 = 1E-14
epsilon = 1E-9
g = 0#9.8#grav*
lam = -0.01#Friction*

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
        F = ((G*self.mass*m2 - C*self.charge*q2)/R2) - (N*m2/(self.mass*R4)) + (N2*torch.where(q2 == self.charge, 1, 0) *
                                                                 torch.where(m2 == self.mass, 1, 0)/R2)  # <- attracting like particles to each other using kronecker delta
        #dr = torch.sum((F*dX/(R2**(1/2)))/self.mass, dim=0) + torch.tensor((lam*self.dx, lam*self.dy + g))
        dr = torch.sum((F*dX/(R2**(1/2)))/(self.mass * (1/FPS**2)), dim=0) + torch.tensor((lam*self.dx, lam*self.dy + g))
        self.dx += dr[0]
        self.dy += dr[1]
        if self.pos[0] > xmax or self.pos[0] < 10:
            self.dx = -self.dx  # /1.01
        if self.pos[1] > ymax or self.pos[1] < 10:
            self.dy = -self.dy  # /1.01
        self.pos[0] += self.dx
        self.pos[1] += self.dy
        return self.dx, self.dy
    
WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

BLACK = (0,0,0)
DAY = (20,20,19)
NIGHT = (10,11,15)

FPS = 60

RADIUS = 15

def draw_window(tensor):
    DISH.fill(BLACK)
    for i in tensor:
        pygame.draw.circle(DISH, [i[2], i[3], i[4]], (i[0], i[1]), RADIUS)
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True 
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        width = torch.randint(1600,(100, 1)).to(dtype=torch.float)
        height = torch.randint(900,(100, 1)).to(dtype=torch.float)
        vx = torch.randint(150,(100, 1)).to(dtype=torch.float)
        vy = torch.randint(150,(100, 1)).to(dtype=torch.float)
        cell_dynamics = torch.cat((width, height, vx, vy), dim = 1)
        color = torch.randint(255, (100, 3))

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

        draw_window(arty)              
    pygame.quit()

if __name__ == '__main__':
    main()


