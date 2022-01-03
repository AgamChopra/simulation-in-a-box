import pygame
import random

WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Evolve')

WHITE = (255,255,255)
BLACK = (0,0,0)
DAY = (20,20,19)
NIGHT = (10,11,15)

FPS = 60

SCALE = (10,10)

SAMPLE_CELL = pygame.image.load(r'E:\evolution\assets\sample_cell.png')
SAMPLE_CELL = pygame.transform.scale(SAMPLE_CELL, SCALE)

def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

def draw_window(x,cell,angle):
    DISH.fill(x)
    for i in range(10,1590,100):
        for j in range(10,890,100):
            blitRotateCenter(DISH, cell, (i,j), angle)
    pygame.display.update()

def main():
    cell = SAMPLE_CELL
    angle = 0
    scene_time = 0
    clock = pygame.time.Clock()
    run = True 
    d_n_ctr = 0
    day_night = DAY
    alive = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False   
       #DAY/NIGHT CYCLE 
        if d_n_ctr < 500:
            d_n_ctr += 1
        else:
            d_n_ctr = 0
            if day_night == DAY:
                day_night = NIGHT
            else: day_night = DAY
        if scene_time > 1000 and random.random() < 0.001:
            cell = pygame.image.load(r'E:\evolution\assets\remains.png')
            cell = pygame.transform.scale(cell, SCALE)
            alive = False
        if alive:
            angle += 20
        draw_window(day_night,cell,angle)
        scene_time += 1
              
    pygame.quit()

if __name__ == '__main__':
    main()