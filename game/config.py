import pygame

# Размеры игры
BLOCK_SIZE = 20
GRID_SIZE = 20
WIDTH = GRID_SIZE * BLOCK_SIZE
HEIGHT = WIDTH

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Скорость игры
SPEED = 40  # FPS

# Направления
DIRECTIONS = {
    (0, -1): 'UP',
    (0, 1): 'DOWN',
    (-1, 0): 'LEFT',
    (1, 0): 'RIGHT'
}

# Действия агента
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_STRAIGHT = 0
