import tkinter as tk


class CharBorad:
    board = []
    is_open = True

    def __init__(self, width, height, on_submit, is_train):
        self.width = width
        self.height = height
        self.on_submit = on_submit
        self.is_train = is_train
        self.__reset_board_()

    def __reset_board_(self):
        self.board = [[False for i in range(self.width)]
                      for i in range(self.height)]

    def __on_exit(self):
        self.is_open = False
        self.window.destroy()

    def __form_output_data(self, raw_inputs):
        final_inputs = []
        for row in raw_inputs:
            final_inputs.extend([1 if item else -1 for item in row])

        return final_inputs

    def start(self):
        # initiale board
        window = tk.Tk()
        window.title("Please drow your char X or O")
        window.eval('tk::PlaceWindow %s center' %
                    window.winfo_pathname(window.winfo_id()))

        self.window = window

        for i in range(self.width):
            for j in range(self.height):
                frame = tk.Frame(
                    master=self.window,
                    relief=tk.RAISED
                )
                frame.grid(row=i, column=j)
                label = tk.Label(
                    master=frame,
                    bg="white",
                    borderwidth=1,
                    width=10,
                    height=5,
                    relief=tk.RAISED,
                )

                label.bind('<Button-1>', lambda e, i=i, j=j,
                           lbl=label: self.on_click(i, j, lbl))
                label.pack()

        btn_clear = tk.Button(
            master=window,
            text="clear",
            command=self.clear_board,
            height=3,
            relief=tk.RAISED,
        )
        btn_clear.grid(row=self.height, column=0, sticky='we')

        if self.is_train:
            btn_x = tk.Button(
                master=window,
                text="X",
                command=lambda: self.submit(1),
                width=9,
                height=3,
                relief=tk.RAISED,
            )
            btn_x.grid(row=self.height, column=1)

            btn_o = tk.Button(
                master=window,
                text="O",
                command=lambda: self.submit(-1),
                width=9,
                height=3,
                relief=tk.RAISED,
            )
            btn_o.grid(row=self.height, column=2)
        else:
            btn_recognize = tk.Button(
                master=window,
                text="recognize",
                command=lambda: self.submit(),
                height=3,
                relief=tk.RAISED,
            )
            btn_recognize.grid(row=self.height, column=1,
                               columnspan=2, sticky='we')

        btn_exit = tk.Button(
            master=window,
            text="exit",
            command=self.__on_exit,
            width=9,
            height=3,
            relief=tk.RAISED,
        )
        btn_exit.grid(row=self.height, column=4)

        self.window.mainloop()
        return

    def clear_board(self):
        self.__reset_board_()

        for item in all_children(self.window):
            if item.widgetName == 'label':
                item.config(bg='white')

    def on_click(self, i, j, label):
        self.board[i][j] = not self.board[i][j]

        label.config(bg='blue' if self.board[i][j] else 'white')

    # x = 1, o = -1
    def submit(self, t=0):
        self.window.destroy()

        if self.is_train:
            self.on_submit(self.__form_output_data(self.board), t)
        else:
            self.on_submit(self.__form_output_data(self.board), None)

        self.__reset_board_()


def all_children(window):   
    _list = window.winfo_children()

    for item in _list:
        if item.winfo_children():
            _list.extend(item.winfo_children())

    return _list


if __name__ == "__main__":
    test = CharBorad(5, 5, lambda inputs, t: print(
        str(inputs)), False)
    test.start()
