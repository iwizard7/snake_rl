import matplotlib.pyplot as plt
from matplotlib.widgets import Button

plt.ion()

qmap_clicked = False
load_clicked = False
save_clicked = False
play_flag = False
pause_flag = False

def on_play(event):
    global play_flag
    play_flag = True

def on_pause(event):
    global pause_flag
    pause_flag = True

def on_load(event):
    global load_clicked
    load_clicked = True

def on_save(event):
    global save_clicked
    save_clicked = True

def on_qmap(event):
    global qmap_clicked
    qmap_clicked = True

def plot(scores, mean_scores, loss_history=None):
    fig = plt.figure('Статистика обучения', figsize=(14, 8))
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

    # Add buttons
    ax_play = fig.add_axes([0.05, 0.01, 0.12, 0.04])
    b_play = Button(ax_play, 'PLAY', color='green')
    b_play.on_clicked(on_play)

    ax_pause = fig.add_axes([0.17, 0.01, 0.12, 0.04])
    b_pause = Button(ax_pause, 'PAUSE', color='red')
    b_pause.on_clicked(on_pause)

    ax_load = fig.add_axes([0.32, 0.01, 0.12, 0.04])
    b_load = Button(ax_load, 'LOAD', color='lightgreen')
    b_load.on_clicked(on_load)

    ax_save = fig.add_axes([0.47, 0.01, 0.12, 0.04])
    b_save = Button(ax_save, 'SAVE', color='orange')
    b_save.on_clicked(on_save)

    ax_map = fig.add_axes([0.62, 0.01, 0.12, 0.04])
    b_map = Button(ax_map, 'Q-MAP', color='lightblue')
    b_map.on_clicked(on_qmap)

    plt.pause(.1)
