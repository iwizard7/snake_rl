import numpy as np

class StateBuilder:
    def __init__(self):
        pass

    def build_state(self, game):
        """
        Вариант B: инженерные признаки

        state = [
            danger_straight,
            danger_left,
            danger_right,
            food_left_rel,
            food_right_rel,
            food_straight,
            dir_l, dir_u, dir_r, dir_d
        ]
        """
        head = game.head

        # Current direction one-hot
        dir_l = game.direction[0] == -1 and game.direction[1] == 0
        dir_u = game.direction[0] == 0 and game.direction[1] == -1
        dir_r = game.direction[0] == 1 and game.direction[1] == 0
        dir_d = game.direction[0] == 0 and game.direction[1] == 1

        # Danger straight
        danger_straight = game.is_collision([head[0] + game.direction[0], head[1] + game.direction[1]])

        # Danger left
        left_dir = (-game.direction[1], game.direction[0])
        danger_left = game.is_collision([head[0] + left_dir[0], head[1] + left_dir[1]])

        # Danger right
        right_dir = (game.direction[1], -game.direction[0])
        danger_right = game.is_collision([head[0] + right_dir[0], head[1] + right_dir[1]])

        # Food in action directions
        straight_pos = [head[0] + game.direction[0], head[1] + game.direction[1]]
        food_straight = straight_pos[0] == game.food[0] and straight_pos[1] == game.food[1]

        left_pos = [head[0] + left_dir[0], head[1] + left_dir[1]]
        food_left_rel = left_pos[0] == game.food[0] and left_pos[1] == game.food[1]

        right_pos = [head[0] + right_dir[0], head[1] + right_dir[1]]
        food_right_rel = right_pos[0] == game.food[0] and right_pos[1] == game.food[1]

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
