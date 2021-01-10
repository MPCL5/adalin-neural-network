import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from adalin import Adalin

data_set = [
    {'inputs': [1, 1], 'result': 1},
    {'inputs': [1, 0], 'result': -1},
    {'inputs': [0, 1], 'result': -1},
    {'inputs': [0, 0], 'result': -1},
]

# data_set = [item for i in range(10) for item in data]


def draw_plot(weights, biace, title='And neural network'):
    w2 = 1
    handler = 0

    if weights[0] != 0:
        w2 = -weights[1] / weights[0]
        biace = -biace / weights[0]
        handler = 1

    dotX = [item['inputs'][0] for item in data_set]
    dotY = [item['inputs'][1] for item in data_set]

    x = np.linspace(-5, 5, 20)
    fig, ax = plt.subplots()  # Create a figure and an axes.

    loc = plticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    ax.plot(handler*x, w2 * x + biace)  # Plot more data on the axes...
    # Plot more data on the axes...
    ax.plot(dotX, dotY, 'o', color='tab:brown')
    ax.set_xlabel('x1')  # Add an x-label to the axes.
    ax.set_ylabel('x2')  # Add a y-label to the axes.
    ax.set_title(title)  # Add a title to the axes.
    ax.grid(linewidth=1, mew=1, ms=1, markevery=1, zorder=1)

    plt.show()


if __name__ == '__main__':
    neural_adalin = Adalin(2, .001, .05)
    neural_adalin.train_all(data_set)

    for item in data_set:
        print(str(item['inputs']) +' expected: ' + str(item['result']) + ' predicted: ' + str(neural_adalin.predict_one(item['inputs'])))

    draw_plot(neural_adalin.get_weights(), neural_adalin.get_bias(),
              f'And ADALIN neural Aplha={neural_adalin.aplpha}')
    input()
