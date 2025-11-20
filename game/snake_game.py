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
        self.grid_size = GRID_SIZE
        self.size_text = str(self.grid_size)
        self.active_field = None
        self.save_model = False
        self.load_model = False
        self.games_count = 0
        self.epsilon_value = 1.0
        self.display = pygame.display.set_mode((self.w, self.h + 80))
        pygame.display.set_caption('SnakeRL')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # RIGHT
        self.head = [self.grid_size // 2, self.grid_size // 2]
        self.snake = [self.head.copy()]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.done = False
        self.paused = False
        self.running = True

    def _place_food(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
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
                elif self.active_field == 'size':
                    if event.key == pygame.K_BACKSPACE:
                        self.size_text = self.size_text[:-1]
                    elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                        if len(self.size_text) < 2:  # max 2 digits
                            self.size_text += chr(event.key - pygame.K_0 + ord('0'))
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        try:
                            new_size = int(self.size_text)
                            if 5 <= new_size <= 50:
                                self.grid_size = new_size
                                self.size_text = str(self.grid_size)
                                print(f'Grid size set to {new_size}')
                                self.reset()
                        except ValueError:
                            pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    x, y = pygame.mouse.get_pos()
                    if 10 <= x <= 90 and HEIGHT + 10 <= y <= HEIGHT + 40:
                        self.paused = False
                    elif 100 <= x <= 180 and HEIGHT + 10 <= y <= HEIGHT + 40:
                        self.paused = True
                    elif 250 <= x <= 310 and HEIGHT + 10 <= y <= HEIGHT + 40:
                        self.active_field = 'size'
                    elif 10 <= x <= 70 and HEIGHT + 45 <= y <= HEIGHT + 75:
                        self.save_model = True
                    elif 80 <= x <= 140 and HEIGHT + 45 <= y <= HEIGHT + 75:
                        self.load_model = True
                    else:
                        self.active_field = None

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

        # Score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        # Training stats
        games_text = button_font.render(f'Games: {self.games_count}', True, WHITE)
        self.display.blit(games_text, (0, 50))
        epsilon_text = button_font.render(f'Epsilon: {self.epsilon_value:.3f}', True, WHITE)
        self.display.blit(epsilon_text, (150, 50))

        # Pause/play buttons
        button_font = pygame.font.SysFont('arial', 16)
        play_color = GREEN if not self.paused else GRAY
        pygame.draw.rect(self.display, play_color, pygame.Rect(10, HEIGHT + 10, 80, 30))
        play_text = button_font.render('PLAY', True, BLACK)
        self.display.blit(play_text, (20, HEIGHT + 15))

        pause_color = RED if self.paused else GRAY
        pygame.draw.rect(self.display, pause_color, pygame.Rect(100, HEIGHT + 10, 80, 30))
        pause_text = button_font.render('PAUSE', True, BLACK)
        self.display.blit(pause_text, (110, HEIGHT + 15))

        if self.paused:
            pause_msg = font.render("PAUSED", True, RED)
            self.display.blit(pause_msg, [0, 30])

        # Input field for grid size
        size_label = button_font.render('Size:', True, WHITE)
        self.display.blit(size_label, (200, HEIGHT + 15))
        pygame.draw.rect(self.display, WHITE, pygame.Rect(250, HEIGHT + 10, 60, 30), 2)
        size_surf = button_font.render(self.size_text, True, WHITE)
        self.display.blit(size_surf, (255, HEIGHT + 15))

        # Save and Load buttons
        save_color = GRAY
        pygame.draw.rect(self.display, save_color, pygame.Rect(10, HEIGHT + 45, 60, 30))
        save_text = button_font.render('SAVE', True, BLACK)
        self.display.blit(save_text, (15, HEIGHT + 50))

        load_color = GRAY
        pygame.draw.rect(self.display, load_color, pygame.Rect(80, HEIGHT + 45, 60, 30))
        load_text = button_font.render('LOAD', True, BLACK)
        self.display.blit(load_text, (85, HEIGHT + 50))

        pygame.display.flip()

        self.clock.tick(SPEED)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt[0] > self.grid_size - 1 or pt[0] < 0 or pt[1] > self.grid_size - 1 or pt[1] < 0:
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
