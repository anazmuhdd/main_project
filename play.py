import pygame
import time

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("50.mp3")
pygame.mixer.music.play()

# Keep script alive while sound is playing
while pygame.mixer.music.get_busy():
    time.sleep(0.1)

