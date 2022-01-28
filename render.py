import pygame
import torch
import physics

WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

BLACK = (0,0,0)
WHITE = (255,255,255)
FPS = 45
RADIUS = 5


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) for cell in arty]
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True 
    
    N = 1000
    
    width = torch.randint(10,WIDTH-10,(N, 1)).to(dtype=torch.float)
    height = torch.randint(10,HEIGHT-10,(N, 1)).to(dtype=torch.float)
    vx = torch.randint(-500,500,(N, 1)).to(dtype=torch.float)
    vy = torch.randint(-500,500,(N, 1)).to(dtype=torch.float)
    ringo = torch.cat((width, height, vx, vy), dim = 1)
    color = torch.randint(30,230, (N, 3))
    illum = torch.randint(0,100,(N,1))

    COLL_DIST = RADIUS * 2
    
    time_step = 0
    
    #lighting = (2,20,70)
    
    Temp = []
    
    S_E = []
    
    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, ringo, sE, temp, n1, n2 = physics.dynamics(ringo, color, time_step/FPS, COLL_DIST=COLL_DIST, RADIUS=RADIUS*1E-6, MASS=1E-13, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF = 1/FPS)
        S_E.append(sE)
        Temp.append(temp)  
        lighting = (int(230*sE/4.2) , int(250*sE/4.2), int(210*sE/4.2)+45)
        arty[:,2:] = (arty[:,2:] * torch.nan_to_num(torch.where(illum < 5, 1., torch.nan),nan=sE/4)).to(dtype = torch.int)
        draw_window(arty,lighting) 
        time_step += 1
    pygame.quit()
    
    physics.plt.plot(Temp,'red') 
    physics.plt.title('Temp')
    physics.plt.show()  
    physics.plt.plot(S_E,'orange')
    physics.plt.title('Solar Energy')
    physics.plt.show()


if __name__ == '__main__':
    main()