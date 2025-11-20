import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    plt.figure('Статистика обучения')
    plt.clf()
    plt.title('Обучение агента RL')
    plt.xlabel('Игры')
    plt.ylabel('Счет')
    plt.plot(scores, label='Индивидуальный счет')
    plt.plot(mean_scores, label='Средний счет')
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(.1)
