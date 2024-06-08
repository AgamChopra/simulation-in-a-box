import pygame
import torch
import physics

WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FPS = 144
RADIUS = 3


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]],
                        (cell[0], cell[1]), RADIUS) for cell in arty]
    pygame.display.update()


def make_cell(gene_type):
    cell = None
    return


def birth_cell(gene1, gene2=None):
    cell = None
    return cell


def main():
    clock = pygame.time.Clock()
    run = True

    N = 1200  # upper limit on threshold.

    width = torch.randint(10, WIDTH-10, (N, 1)).to(dtype=torch.float)
    height = torch.randint(10, HEIGHT-10, (N, 1)).to(dtype=torch.float)

    # transfer over to cell function. updated every step.
    vx = torch.randint(-500, 500, (N, 1)).to(dtype=torch.float)
    # transfer over to cell function. updated every step.
    vy = torch.randint(-500, 500, (N, 1)).to(dtype=torch.float)

    ringo = torch.cat((width, height, vx, vy), dim=1)

    # transfer to genes. updated at birth.
    color = torch.randint(30, 250, (N, 3))
    # transfer to genes. updated at birth.
    illum = torch.randint(0, 100, (N, 1))

    COLL_DIST = RADIUS * 2

    time_step = 0

    Temp = []

    S_E = []

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        arty, ringo, sE, temp, n1, n2 = physics.dynamics(
            ringo, color, time_step/FPS, COLL_DIST=COLL_DIST, RADIUS=RADIUS*1E-6, MASS=1E-13, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF=1.67E-2)
        S_E.append(sE)
        Temp.append(temp)
        lighting = (int(230*sE/4.2), int(250*sE/4.2), int(210*sE/4.2)+45)
        arty[:, 2:] = (arty[:, 2:] * torch.nan_to_num(torch.where(illum <
                       5, 1., torch.nan), nan=sE/4.2)).to(dtype=torch.int)
        draw_window(arty, lighting)
        time_step += 1
    pygame.quit()

    physics.plt.plot(Temp, 'red')
    physics.plt.title('Temp')
    physics.plt.show()
    physics.plt.plot(S_E, 'orange')
    physics.plt.title('Solar Energy')
    physics.plt.show()


# ringo (N,4) (:,:2)-> x,y positions (:,2:)-> x,y velocity. ringo passed to the physics engine.
# arty (N,6) -> x,y positions , r,g,b colors, i illumination
# Step 1: Initialize initial conditions
# Step 2: cells interact with environment and do work based on neurons associated with genes... colors are updated for arty in Step 4.
# Step 3: ringo is updated based on Step 2...
# Step 4: physics ia applied based on ringo and arty is updated based on Step 3 and 2...
# Step 5: loop over step 2 to 4...
class environment():
    def __init__(self, pop_cap=1000, start_pop=50):
        SOLAR, TEMP, N_1, N_2 = 0, 0, 0, 0
        # All allive cells will have a unique id. after death cell turns into N1, N2, and TEMP...
        self.objects = {'Environment': [SOLAR, TEMP, N_1, N_2]}
        for i in range


if __name__ == '__main__':
    main()
