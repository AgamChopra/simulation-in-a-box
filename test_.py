import pygame
import random
import torch

C = 8.98755E9
G = 6.67408E-11
N = 1E-14
N2 = 1E-14
epsilon = 1E-9
g = 9.8#grav*
lam = -0.05#Friction*

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

FPS = 144

RADIUS = 15

K = 100
K_list = range(K)

def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

def draw_window(op,q2,m2):
    DISH.fill(BLACK)    
    d = torch.tensor([[op[i].pos[0], op[i].pos[1]] for i in K_list])
    for i in K_list:
        global_info = [torch.cat([d[:i],d[i:]]), torch.cat([m2[:i],m2[i:]]), torch.cat([q2[:i],q2[i:]])] if i < K-1 else [d[:i], m2[:i], q2[:i]]
        op[i].update(WIDTH-10, HEIGHT-10, global_info)
        pygame.draw.circle(DISH, [random.randrange(255), random.randrange(255), random.randrange(255)], (int(op[i].pos[0]),int(op[i].pos[1])), RADIUS)
    pygame.display.update()


def color(val):
    s = str(val)
    tup = (s[:3], s[3:6], s[6:])
    return list(map(lambda x: int(x), tup))

def main():
    clock = pygame.time.Clock()
    run = True 
    op = []
    charge = [1E-9,-1E-9]
    for i in K_list:
        if i < 0.99*K:
            [op.append(cell_physics(charge[0], 9E-8, [random.randint(10, WIDTH-10), random.randint(10, HEIGHT-10)], random.randint(-10, 10)*0, random.randint(-10, 10)*0))]
        else: [op.append(cell_physics(0, 7E-8, [WIDTH/2, HEIGHT/2], random.randint(-1, 1)*0, random.randint(-1, 1)*0))]
    q2 = [op[i].charge for i in range(K)]
    q2 = torch.tensor((q2)).unsqueeze(1)
    m2 = [op[i].mass for i in range(K)]
    m2 = torch.tensor((m2)).unsqueeze(1)
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False   
        draw_window(op,q2,m2)              
    pygame.quit()

if __name__ == '__main__':
    main()


