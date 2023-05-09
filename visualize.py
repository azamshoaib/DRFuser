from video import make_video
import numpy as np
import pandas as pd
import pygame
import glob
import cv2
import matplotlib.pyplot as plt
from pygame.locals import *
import os

pause=0

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)


file_name = pd.read_csv('path to csv file')


pygame.init()
size = (1024, 256)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)
video_folder = '/home/shoaib/work/research/paper_projects/Fusion_dvs_rgb_ss/video_folder/'
os.mkdir(video_folder)
save_screen = make_video(video_folder,screen)
video = True

# for i in range(100):
for i in range(len(file_name)):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pygame.time.delay(2000)
                print("Pausing!")
            if event.key == pygame.K_1:
                pygame.time.delay(0)
                print("Resuming!")
                continue
                

    pred_angle = file_name['Predicted_angle'].iloc[i] # radians
    true_angle = file_name["True_angle"].iloc[i]
    # rgb_angle = file_name_r['Predicted_angle'].iloc[i] # radians
    # dvs_angle = file_name_d['Predicted_angle'].iloc[i] # radians
    dvs_img = pygame.image.load(file_name['DVS_filename'].iloc[i])
    screen.blit(dvs_img, (512, 0))
    # add image to screen
    img = pygame.image.load(file_name['APS_filename'].iloc[i])
    # img = pygame.transform.scale(img, (346, 260))
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(pred_angle*57.2958, 3)), 1, (255,0,0),(0,0,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle*57.2958, 3)), 1, (255,255,255),(0,0,0)) # angle in degrees
    # rgb_txt = myfont.render("RGB angle:" + str(round(rgb_angle*57.2958, 3)), 1, (0,140,255)) # angle in degrees
    # dvs_txt = myfont.render("DVS angle:" + str(round(dvs_angle*57.2958, 3)), 1, (140,25,255)) # angle in degrees
    # true_txt_rad = myfont.render("True angle_Rad:" + str(round(angle, 4)), 1, (255,255,255)) # angle in rad
    screen.blit(pred_txt, (10, 10))
    screen.blit(true_txt, (10, 30))
    # screen.blit(rgb_txt, (10, 50))
    # screen.blit(dvs_txt, (10, 70))
    # screen.blit(true_txt_rad, (10, 30))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [250, 220], radius, 2) 
    pygame.draw.circle(screen, WHITE, [780, 220], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [250 + int(x), 220- int(y)], 7)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + pred_angle)
    y = radius * np.sin(np.pi/2 + pred_angle)
    pygame.draw.circle(screen, RED, [250 + int(x), 220 - int(y)], 5) 

    # # # draw circle for rgb angle
    # x_rgb = radius * np.cos(np.pi/2 + rgb_angle)
    # y_rgb = radius * np.sin(np.pi/2 + rgb_angle)
    # pygame.draw.circle(screen, BLUE, [200 + int(x_rgb), 200 - int(y_rgb)], 5) 

    # draw cricle for true angle----DVS
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [780 + int(x), 220 - int(y)], 7)
    
    # draw cricle for predicted angle----DVS
    x = radius * np.cos(np.pi/2 + pred_angle)
    y = radius * np.sin(np.pi/2 + pred_angle)
    pygame.draw.circle(screen, RED, [780+ int(x), 220 - int(y)], 5) 
    

    # # draw circle for dvs angle
    # x_dvs = radius * np.cos(np.pi/2 + dvs_angle)
    # y_dvs = radius * np.sin(np.pi/2 + dvs_angle)
    # pygame.draw.circle(screen, GREEN, [690+ int(x_dvs), 200 - int(y_dvs)], 5) 

    #pygame.display.update()
    pygame.display.flip()
    next(save_screen)
    pygame.time.delay(20)