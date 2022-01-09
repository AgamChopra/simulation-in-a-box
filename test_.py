import pygame
import torch
import physics

WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

BLACK = (0,0,0)
FPS = 60
RADIUS = 5


def draw_window(tensor):
    DISH.fill(BLACK)
    for cell in tensor:
        pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS)
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True 
    
    N = 5000
    
    width = torch.randint(1600,(N, 1)).to(dtype=torch.float)
    height = torch.randint(900,(N, 1)).to(dtype=torch.float)
    vx = torch.randint(-500,500,(N, 1)).to(dtype=torch.float)
    vy = torch.randint(-500,500,(N, 1)).to(dtype=torch.float)
    cell_dynamics = torch.cat((width, height, vx, vy), dim = 1)
    color = torch.randint(255, (N, 3))

    COLL_DIST = RADIUS * 2
    
    time_step = 0
    
    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, cell_dynamics = physics.dynamics(cell_dynamics, color, time_step, COLL_DIST=COLL_DIST, RADIUS=RADIUS, MASS=1E-6, WIDTH=WIDTH, HEIGHT=HEIGHT)
        draw_window(arty) 
        time_step += 1
    pygame.quit()


if __name__ == '__main__':
    main()