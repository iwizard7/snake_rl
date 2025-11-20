import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

plt.ion()

def show_q_heatmap(agent, grid_size=20, fixed_dir=0):
    """
    Показать heatmap Q-значений для NN агента на сетке grid_size x grid_size
    fixed_dir: фиксированное направление (0: left, 1: up, 2: right, 3: down)
    """

    # Создать сетку состояний
    q_map = np.zeros((grid_size, grid_size, 3))  # Q для 3 действий

    # Фиксированная еда в центре
    food = [grid_size//2, grid_size//2]

    directions = {
        0: (-1, 0),  # LEFT
        1: (0, -1),  # UP
        2: (1, 0),   # RIGHT
        3: (0, 1),   # DOWN
    }

    curr_direction = directions[fixed_dir]

    for x in range(1, grid_size-1):  # Избегать краев чтобы не было столкновений
        for y in range(1, grid_size-1):

            head = [x, y]

            # Вычислить состояние вручную (повторяя логику StateBuilder)

            # Danger checks
            danger_straight = (x + curr_direction[0] < 0 or x + curr_direction[0] >= grid_size or
                             y + curr_direction[1] < 0 or y + curr_direction[1] >= grid_size)

            left_dir = (-curr_direction[1], curr_direction[0])  # поворот налево
            danger_left = (x + left_dir[0] < 0 or x + left_dir[0] >= grid_size or
                          y + left_dir[1] < 0 or y + left_dir[1] >= grid_size)

            right_dir = (curr_direction[1], -curr_direction[0])  # поворот направо
            danger_right = (x + right_dir[0] < 0 or x + right_dir[0] >= grid_size or
                           y + right_dir[1] < 0 or y + right_dir[1] >= grid_size)

            # Food direction relative to movement
            straight_pos = [x + curr_direction[0], y + curr_direction[1]]
            food_straight = straight_pos == food

            left_pos = [x + left_dir[0], y + left_dir[1]]
            food_left = left_pos == food

            right_pos = [x + right_dir[0], y + right_dir[1]]
            food_right = right_pos == food

            # Direction one-hot
            dir_l = curr_direction == (-1, 0)
            dir_u = curr_direction == (0, -1)
            dir_r = curr_direction == (1, 0)
            dir_d = curr_direction == (0, 1)

            state = [
                float(danger_straight),      # danger straight
                float(danger_left),          # danger left
                float(danger_right),         # danger right
                float(food_left),            # food left
                float(food_right),           # food right
                float(food_straight),        # food straight
                float(dir_l),                # direction left
                float(dir_u),                # direction up
                float(dir_r),                # direction right
                float(dir_d),                # direction down
            ]

            # Получить Q-значения
            state_tensor = torch.tensor(state, dtype=torch.float).to(agent.trainer.model.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.model(state_tensor).cpu().numpy().flatten()

            q_map[x, y] = q_values

    # Найти предпочитаемые действия
    decision_map = np.argmax(q_map, axis=2).astype(float)
    # Обнулить неиспользуемые ячейки
    for x in range(grid_size):
        if x < 1 or x >= grid_size-1:
            decision_map[x, :] = np.nan
        for y in range(grid_size):
            if y < 1 or y >= grid_size-1:
                decision_map[x, y] = np.nan

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Преимущественные действия
    ax = axes[0]
    cmap = plt.cm.get_cmap('Set1', 3)
    im = ax.imshow(decision_map.T, origin='lower', cmap=cmap, vmin=0, vmax=2)
    ax.set_title(f'Преимущественные действия (напр. {["←", "↑", "→", "↓"][fixed_dir]})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Colorbar для действий
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
    cbar.set_ticklabels(['Вперед', 'Налево', 'Направо'])
    cbar.set_label('Действие')

    # Q-значения для каждого действия
    actions = ['Вперед', 'Налево', 'Направо']
    cmaps = ['Blues', 'Reds', 'Greens']

    for i in range(3):
        ax = axes[i+1]
        data = q_map[:, :, i].T
        # Обнулить края
        data[:, 0] = np.nan
        data[:, -1] = np.nan
        data[0, :] = np.nan
        data[-1, :] = np.nan

        im = ax.imshow(data, origin='lower', cmap=cmaps[i])
        ax.set_title(f'Q-{actions[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()

    return fig

# Для вызова в main
if __name__ == "__main__":
    # Пример
    show_q_heatmap(None, 20, 0)
