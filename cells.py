import random
from graphics import *
import sys
sys.path.append('E:\evolution')
#from numba import jit, cuda, vectorize


class cell():

    def __init__(self, screen, position=Point(450, 450), color=[20, 200, 100], size=5):
        self.cell = Circle(center=position, radius=size)
        self.cell.setFill(color_rgb(color[0], color[1], color[2]))
        self.screen = screen
        self.cell.draw(screen)

    def update(self, dx, dy, color=None):
        self.cell.move(dx, dy)
        if color != None:
            self.cell.setFill(color_rgb(color[0], color[1], color[2]))


def setup_environment(height=600, width=600):
    screen = GraphWin("Cell Evolution", width, height)
    screen.setBackground(color_rgb(0, 0, 0))
    center = Point(int(height/2), int(width/2))
    # screen.getKey()
    # screen.close()
    return screen, center


def example(execute=False):

    print('\nscreen,_ = setup_environment(900,1600)')
    print('o = []')
    print('[o.append(cell(screen,Point(random.randint(0, 1600), random.randint(0, 900)),[random.randint(0, 250), random.randint(0, 250), random.randint(0, 250)],random.randint(5, 20))) for _ in range(5)]')
    print('screen.getKey()')
    print('k=0.05\n')
    print('while True:')
    print('    if(len(o)<=500): ')
    print(
        '         o.append(cell(screen,Point(random.randint(0, 1600), random.randint(0, 900)),[random.randint(0, 250), random.randint(0, 250), random.randint(0, 250)],random.randint(5, 20)))')
    print('    time.sleep(k)')
    print(
        '    [obj.update(random.randint(-5, 5), random.randint(-5, 5)) for obj in o]')
    print('    time.sleep(k)')
    print('\n#screen.getKey()')
    print('#screen.close()')

    if execute:
        screen, _ = setup_environment(height=900, width=1600)
        o = []
        [o.append(cell(screen, Point(random.randint(0, 1600), random.randint(0, 900)), [random.randint(
            0, 250), random.randint(0, 250), random.randint(0, 250)], random.randint(5, 20))) for _ in range(5)]
        screen.getKey()
        k = 0.05

        while True:
            if(len(o) <= 100):
                o.append(cell(screen, Point(random.randint(0, 1600), random.randint(0, 900)), [random.randint(
                    0, 250), random.randint(0, 250), random.randint(0, 250)], random.randint(5, 20)))
            time.sleep(k)
            [obj.update(random.randint(-5, 5), random.randint(-5, 5))
             for obj in o]
            time.sleep(k)
# example(True)
