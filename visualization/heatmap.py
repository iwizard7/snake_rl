import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

plt.ion()

def show_q_heatmap(agent, grid_size=20, fixed_dir=0):
    """
    Показать heatmap Q-значений для агента на сетке grid_size x grid_size
    fixed_dir: фиксированное направление (0: left, 1: up/te, 2: right, 3: down)
    """
    
    # Создать сетку состояний
    state_builder = agent.state_builder
    q_map = np.zeros((grid_size, grid_size, 3))  # Q для 3 действий
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for x in range(grid_size):
        for y in range(grid_size):
            # Фиксированная голова змейки на (x,y), пустая (нет тела)
            fake_game = {
                'head': [x, y],
                'direction': [(fixed_dir == 0), 0, (fixed_dir == 2), (fixed_dir == 1)],  # one-hot dir
                'snake': [[x, y]],  # только голова
                'food': [grid_size//2, grid_size//2]  # центр
            }
            
            # Получить Q-значения
            state = state_builder.build_state(fake_game)
            state_tensor = torch.tensor(state, dtype=torch.float).to(agent.trainer.model.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.model(state_tensor).cpu().numpy().flatten()
            
            q_map[x, y] = q_values
    
    # Отобразить
    actions = ['Вперед', 'Налево', 'Направо']
    
    vmin = q_map.min()
    vmax = q_map.max()
    
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(q_map[:, :, i].T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Q-{actions[i]} (напр. {["Л", "В", "Н", "П"][fixed_dir]})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.invert_yaxis()  # Чтобы Y увеличивался вверх
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Q-значение')
    
    plt.tight_layout()
    
    return fig

# Для вызова в main
if __name__ == "__main__":
    # Пример
    show_q_heatmap(None, 20, 0)
