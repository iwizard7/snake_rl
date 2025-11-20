import pygame
import random
import numpy as np
from game.config import *

pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeRL')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # RIGHT
        self.head = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.snake = [self.head.copy()]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.done = False
        self.paused = False
        self.running = True

    def _place_food(self):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def process_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused

    def step(self, action):
        self.frame_iteration += 1
        # 1. Get state
        state = self._get_state()

        # 2. Perform action
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check collision
        reward = 0
        if self.is_collision(self.head) or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            self.done = True
            return state, reward, self.done

        # 4. Eat food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            # No pop when eating
        else:
            self.snake.pop()

        # Shaping reward: moving closer or farther from food
        if not self.done:
            reward += self._calculate_shaping_reward()

        # 5. Update ui
        self.render()

        return state, reward, self.done

    def render(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw snake head in different color
        pygame.draw.rect(self.display, BLUE, pygame.Rect(self.head[0]*BLOCK_SIZE, self.head[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        if self.paused:
            pause_text = font.render("PAUSED - Press P to Play", True, WHITE)
            self.display.blit(pause_text, [0, 30])
        pygame.display.flip()

        self.clock.tick(SPEED)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt[0] > GRID_SIZE - 1 or pt[0] < 0 or pt[1] > GRID_SIZE - 1 or pt[1] < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # action is int: 0 straight, 1 left relative, 2 right
        clock_wise = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # LEFT, UP, RIGHT, DOWN
        idx = clock_wise.index(self.direction)

        if action == ACTION_STRAIGHT:  # straight
            new_dir = clock_wise[idx]
        elif action == ACTION_RIGHT:  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        elif action == ACTION_LEFT:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head[0] + self.direction[0]
        y = self.head[1] + self.direction[1]
        self.head = [x, y]

    def _get_state(self):
        head = self.head

        # Current direction one-hot: left, up, right, down
        dir_l = self.direction[0] == -1 and self.direction[1] == 0
        dir_u = self.direction[0] == 0 and self.direction[1] == -1
        dir_r = self.direction[0] == 1 and self.direction[1] == 0
        dir_d = self.direction[0] == 0 and self.direction[1] == 1

        # Danger straight
        danger_straight = self.is_collision([head[0] + self.direction[0], head[1] + self.direction[1]])

        # Danger left (90 degrees left)
        left_dir = (-self.direction[1], self.direction[0])
        danger_left = self.is_collision([head[0] + left_dir[0], head[1] + left_dir[1]])

        # Danger right
        right_dir = (self.direction[1], -self.direction[0])
        danger_right = self.is_collision([head[0] + right_dir[0], head[1] + right_dir[1]])

        # Food location relative - simple
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]

        # But to make relative, perhaps calculate for each action direction
        # For straight
        straight_pos = [head[0] + self.direction[0], head[1] + self.direction[1]]
        food_straight = straight_pos[0] == self.food[0] and straight_pos[1] == self.food[1]

        # For left turn
        left_turn_dir = (-self.direction[1], self.direction[0])
        left_pos = [head[0] + left_turn_dir[0], head[1] + left_turn_dir[1]]
        food_left_rel = left_pos[0] == self.food[0] and left_pos[1] == self.food[1]

        # For right turn
        right_turn_dir = (self.direction[1], -self.direction[0])
        right_pos = [head[0] + right_turn_dir[0], head[1] + right_turn_dir[1]]
        food_right_rel = right_pos[0] == self.food[0] and right_pos[1] == self.food[1]

        state = [
            int(danger_straight),
            int(danger_left),
            int(danger_right),
            int(food_left_rel),
            int(food_right_rel),
            int(food_straight),
            int(dir_l),
            int(dir_u),
            int(dir_r),
            int(dir_d)
        ]

        return np.array(state, dtype=float)

    def _calculate_shaping_reward(self):
        # Approximate distance before and after move
        dist_before = np.linalg.norm(np.array(self.snake[0]) - np.array(self.food))
        dist_after = np.linalg.norm(np.array(self.head) - np.array(self.food))
        reward = 0.1 if dist_after < dist_before else -0.1
        return reward
