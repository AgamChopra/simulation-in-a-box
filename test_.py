import pygame
import torch

WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

BLACK = (0,0,0)
FPS = 60
RADIUS = 15


def draw_window(tensor):
    DISH.fill(BLACK)
    for cell in tensor:
        pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS)
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
        a = cell_dynamics[:,:2]
        dist = torch.cdist(a, a) # sus
        v_col = torch.where(dist > COLL_DIST, 1., 0.)
        v_col = torch.sum(v_col,dim=1)
        theta = v.shape[0] - 1
        v_col = torch.where(v_col < theta, 0., 1.)
        v = v * v_col.view(v.shape[0],1)
        SPF = 1 # 1 sec/frame
        dx = v * SPF
        pos = (a + dx).to(dtype = torch.int32)

        arty = torch.cat((pos,color),dim = 1)
        draw_window(arty)              
    pygame.quit()

if __name__ == '__main__':
    main()


