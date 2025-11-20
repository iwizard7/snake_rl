import matplotlib.pyplot as plt

plt.ion()

qmap_clicked = False
load_clicked = False
save_clicked = False
play_flag = False
pause_flag = False

def on_click(event):
    # Map coordinates to figure coordinates
    if event.inaxes is None and event.y < 0.1:
        # In the bottom margin
        if 0.0 < event.x < 0.1:
            global play_flag
            play_flag = True
        elif 0.1 < event.x < 0.2:
            global pause_flag
            pause_flag = True
        elif 0.25 < event.x < 0.35:
            global load_clicked
            load_clicked = True
        elif 0.4 < event.x < 0.5:
            global save_clicked
            save_clicked = True
        elif 0.55 < event.x < 0.65:
            global qmap_clicked
            qmap_clicked = True

def plot(scores, mean_scores, loss_history=None):
    fig = plt.figure('Статистика обучения')
    plt.clf()

    if loss_history and len(loss_history) > 0:
        plt.subplot(2, 1, 1)
    plt.title('Обучение агента RL')
    plt.xlabel('Игры')
    plt.ylabel('Счет')
    plt.plot(scores, label='Индивидуальный счет')
    plt.plot(mean_scores, label='Средний счет')
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True)

    if loss_history and len(loss_history) > 0:
        plt.subplot(2, 1, 2)
        plt.plot(loss_history, color='red', label='Потеря модели')
        plt.xlabel('Батчи обучения')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    # Add buttons at bottom
    plt.subplots_adjust(bottom=0.15)
    plt.text(0.05, 0.05, 'PLAY', fontsize=12, ha='center', va='center', transform=fig.transFigure,
             bbox=dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.text(0.15, 0.05, 'PAUSE', fontsize=12, ha='center', va='center', transform=fig.transFigure,
             bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.text(0.3, 0.05, 'LOAD', fontsize=12, ha='center', va='center', transform=fig.transFigure,
             bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.text(0.45, 0.05, 'SAVE', fontsize=12, ha='center', va='center', transform=fig.transFigure,
             bbox=dict(facecolor='orange', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.text(0.6, 0.05, 'Q-MAP', fontsize=12, ha='center', va='center', transform=fig.transFigure,
             bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.pause(.1)
