import numpy as np
import json
from math import sqrt
import random


def digit_shuffle(digit, shuffle_count):
    shuffle_indexes = []
    while shuffle_count > 0:
        new_index = random.randint(0, digit.size-1)
        if new_index not in shuffle_indexes:
            shuffle_indexes.append(new_index)
            digit[new_index] = (digit[new_index] + 1) % 2
            shuffle_count -= 1
    return digit


def print_digit(digit):
    for i in range(10):
        for k in range(10):
            print('#' if digit[i*10+k] else ' ', end='')
        print()


def generate_data_set(amount, clear_digits, lower_shuffle_limit, upper_shuffle_limit):
    data_set = []
    while amount > 0:
        chosen_digit = random.randint(0, 9)
        new_data = clear_digits[chosen_digit].copy()
        new_data = digit_shuffle(new_data, random.randint(lower_shuffle_limit, upper_shuffle_limit))
        data_set.append((new_data, chosen_digit))
        amount -= 1
    return data_set


def approx(input_data, weights):
    result = 0
    for i in range(len(input_data)):
        result += input_data[i]*weights[i]
    result += weights[-1]
    return result


def approxError(edu_set, weights, neuron_digit):
    result_error = 0
    for digit in edu_set:
        result_error += (approx(digit[0], weights) - int(digit[1] == neuron_digit)) ** 2
    return result_error


def gradient(edu_set, weights, neuron_digit, h=1e-05):
    grad = []
    for i, _ in enumerate(weights):
        new_weights = [x_j + (h if j == i else 0) for j, x_j in enumerate(weights)]
        grad.append((approxError(edu_set, new_weights, neuron_digit) - approxError(edu_set, weights, neuron_digit)) / h)
    return grad


class LayerOneNeuron:
    def __init__(self, weight_count, neuron_digit, step_size):
        self.weights = [0 for i in range(weight_count)]
        self.neuron_digit = neuron_digit
        self.step_size = step_size

    def educate(self, edu_set):
        gradients = gradient(edu_set, self.weights, self.neuron_digit)
        G = sqrt(sum(grad ** 2 for grad in gradients))
        while abs(G) > 2:
            print(f"Нейрон {self.neuron_digit} G: {abs(G)}")
            new_weights = [weight - (self.step_size * grad / G) for weight, grad in zip(self.weights, gradients)]
            new_gradients = gradient(edu_set, new_weights, self.neuron_digit)
            new_G = sqrt(sum(grad ** 2 for grad in new_gradients))
            if new_G >= G:
                self.step_size /= 1.2
            else:
                self.weights = new_weights
                gradients = new_gradients
                G = new_G
        print(f"Нейрон {self.neuron_digit} обучен успешно")

    def get_result(self, test_set):
        return [approx(digit[0], self.weights) for digit in test_set]


zero = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

one = np.array([0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1])

two = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

three = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

four = np.array([1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1])

five = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

six = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,0,0,0,0,0,0,0,0]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

seven = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1])

eight = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

nine = np.array([1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [0,0,0,0,0,0,0,0,1,1]+
               [1,1,1,1,1,1,1,1,1,1]+
               [1,1,1,1,1,1,1,1,1,1])

clear_digits = [zero, one, two, three, four, five, six, seven, eight, nine]
edu_data_set = [(digit_shuffle(digit, random.randint(0, 0)), num) for num, digit in enumerate(clear_digits)]
edu_data_set += generate_data_set(20, clear_digits, 1, 2)
test_data_set = generate_data_set(10000, clear_digits, 4, 5)
neuron_list = [LayerOneNeuron(101, i, 0.1) for i in range(10)]
result_list = []
file_index = 0
for neuron in neuron_list:
    try:
        file = open(f"weights/weight{file_index}.bin", "r")
        neuron.weights = json.load(file)
        file.close()
    except IOError:
        print(f"weights/weight{file_index}.bin не найден, обучаем заново")
        neuron.educate(edu_data_set)
        file = open(f"weights/weight{file_index}.bin", "w")
        json.dump(neuron.weights, file)
        file.close()
    file_index += 1

for index,neuron in enumerate(neuron_list):
    result_list.append(neuron.get_result(test_data_set))

good_pred = 0
bad_pred = 0
for i in range(len(result_list[0])):
    max_prob = 0
    prob_sum = 0
    chosen_digit = -1
    for k in range(len(result_list)):
        prob_sum += result_list[k][i]
        if result_list[k][i] > max_prob:
            chosen_digit = k
            max_prob = result_list[k][i]
    if test_data_set[i][1] != chosen_digit:
        bad_pred += 1
        print(f"Image:{test_data_set[i][1]} | Result:{chosen_digit} | Probability:{max_prob * 100}%")
        print_digit(test_data_set[i][0])
        print()
    else:
        good_pred += 1

print(f"Точность: {(good_pred / (good_pred + bad_pred)) * 100}%")

