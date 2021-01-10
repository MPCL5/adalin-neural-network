from adalin import Adalin
from char_board import CharBorad
from load_from_folder import LoadAndSaveData

data_set = []
is_load_from_file = True


def on_submit(inputs, x):
    data_set.append({'inputs': inputs, 'result': x})


def get_char_name(number):
    if number == -1:
        return 'O'
    elif number == 1:
        return 'X'
    else:
        return 'undefined'


if __name__ == "__main__":
    adalin_neuron = Adalin(5*5, .02, .005)  # initialize neuron
    storage = LoadAndSaveData('./single_data_set')

    if is_load_from_file:
        data_set = storage.load_data_set()
    else:
        train_board = CharBorad(5, 5, on_submit, True)  # initialize char board
        while train_board.is_open:
            train_board.start()
        storage.save_data_set()  # save your input data

    adalin_neuron.train_all(data_set)

    print(adalin_neuron.weights)

    board = CharBorad(5, 5,
                      lambda inputs, x: print(get_char_name(adalin_neuron.predict_one(inputs))), False)  # initialize char board
    while board.is_open:
        board.start()
