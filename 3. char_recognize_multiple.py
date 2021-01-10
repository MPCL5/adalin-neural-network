from adalin import Adalin
from char_board import CharBorad
from load_from_folder import LoadAndSaveData

data_set = []
is_load_from_file = True


def on_submit(inputs, x):
    data_set.append({'inputs': inputs, 'result': x})
    

def get_char_name(number_x, number_o):
    if number_x == 1 and not number_o == 1:
        return 'X'
    elif number_o == 1 and not number_x == 1:
        return 'O'
    else:
        return 'Undefined'


if __name__ == "__main__":
    # initialize storage manager
    storage = LoadAndSaveData('./single_data_set')

    if is_load_from_file:
        data_set = storage.load_data_set()
    else:
        # initialize char board
        board = CharBorad(5, 5, on_submit, True)
        while board.is_open:
            board.start()
        storage.save_data_set()  # save your input data

    # initialize neurons
    adalin_neuron_x = Adalin(5*5, .02, .005)
    adalin_neuron_o = Adalin(5*5, .02, .005)

    adalin_neuron_x.train_all(data_set)
    adalin_neuron_o.train_all(
        [{'inputs': item['inputs'], 'result': 1 if item['result'] == -1 else -1} for item in data_set])

    board = CharBorad(5, 5,
                      lambda inputs, x: print(get_char_name(adalin_neuron_x.predict_one(inputs), adalin_neuron_o.predict_one(inputs))), False)  # initialize char board
    while board.is_open:
        board.start()
