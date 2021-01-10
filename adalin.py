class Adalin:
    def __init__(self, n, min_delta, alpha):
        # step.1,2
        self.min_delta = min_delta
        self.aplpha = alpha
        self.n = n
        self.bias = 0
        self.weights = [0 for i in range(n)]

    def __calculate_new_weight(self, old, delta):
        return old + delta

    def __calculate_delta(self, t, y_ni, x):
        return self.aplpha * (t - y_ni) * x

    def __train_one(self, inputs, t):
        # Step.6
        y_ni = self.calculate_one(inputs)
        max_delta = abs(self.__calculate_delta(t, y_ni, 1))
        self.bias = self.__calculate_new_weight(self.bias, self.__calculate_delta(t, y_ni, 1))

        for i in range(len(inputs)):
            # Step.7
            delta = self.__calculate_delta(t, y_ni, inputs[i])
            self.weights[i] = self.__calculate_new_weight(self.weights[i], delta)

            if max_delta == None or abs(delta) > max_delta:
                max_delta = abs(delta)

        return max_delta

    def active_function(self, y_ni):
        """
            active function
        """
        return y_ni

    def quantizer_function(self, y_ni):
        return 1 if y_ni >= 0 else -1

    def calculate_one(self, inputs):
        y_ni = 0
        # calculat xi*wi
        for i in range(len(self.weights)):
            # print(str(self.weights[i]) + ' ' + str(inputs[i]))
            y_ni += self.weights[i] * inputs[i]

        y_ni += self.bias
        return self.active_function(y_ni)

    def predict_one(self, inputs):
        return self.quantizer_function(self.calculate_one(inputs))

    def train_all(self, cases):
        flag = True
        # Step.3
        while flag:
            # Step.4,5
            for case in cases:
                print(str(case))
                # Step.8
                if self.__train_one(case['inputs'], case['result']) < self.min_delta:
                    flag = False
                    break

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias
