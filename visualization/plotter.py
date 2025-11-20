import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores, loss_history=None):
    plt.figure('Статистика обучения')
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

    plt.tight_layout()
    plt.pause(.1)
