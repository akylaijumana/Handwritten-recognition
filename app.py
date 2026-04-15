import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

IMAGESAVE = False
MODEL = load_model('bestmodel.h5')
LABELS = {0: 'zero', 1: 'one',
          2: 'two', 3: 'three',
          4: 'four', 5: 'five',
          6: 'six', 7: 'seven',
          8: 'eight', 9: 'nine'}
BOUNDRYINC = 10
# Initialize the pygame
pygame.init()
FONT = pygame.font.Font('freesansbold.ttf', 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Digit Board')
iswriting = False

Number_xcord = []
Number_ycord = []

image_cnt = 1

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, white, (xcord, ycord), 4, 0)
            Number_xcord.append(xcord)
            Number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            
            if len(Number_xcord) > 0 and len(Number_ycord) > 0:
                Number_xcord = sorted(Number_xcord)
                Number_ycord = sorted(Number_ycord)

                rect_min_x, rect_max_x = max(Number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX,
                                                                                   Number_xcord[-1] + BOUNDRYINC)
                rect_min_Y, rect_max_Y = max(Number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEY,
                                                                                   Number_ycord[-1] + BOUNDRYINC)

                Number_xcord = []
                Number_ycord = []

                ing_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(
                    np.float32)

                if IMAGESAVE:
                    cv2.imwrite("image.png", ing_arr.astype(np.uint8))
                    image_cnt += 1

                if PREDICT:
                    image = cv2.resize(ing_arr, (28, 28))
                    image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255

                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                    textsurface = FONT.render(label, True, red, white)
                    textRecObj = textsurface.get_rect()
                    textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y

                    DISPLAYSURF.blit(textsurface, textRecObj)
            else:
                Number_xcord = []
                Number_ycord = []

        if event.type == KEYDOWN:
            if event.key == K_n:
                DISPLAYSURF.fill(black)

    pygame.display.update()

