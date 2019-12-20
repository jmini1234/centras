""" Neural Network XOR Problem """

import numpy as np


def sigmoid(g):
    return 1 / (1 + np.exp(-2 * g)) 


def sigmoid_gradient(g):
    return g * (1 - g)


def feedForwardProp(input_layer, output_layer, hidden_weights, output_weights, bias):
    z2 = np.dot(input_layer, hidden_weights)
    a2 = sigmoid(z2)
    a2 = a2.T
    a2 = np.vstack((a2, bias)).T
    z3 = np.dot(a2, output_weights)
    a3 = sigmoid(z3)
    return a2, a3, hidden_weights, output_weights


def backPropogation(input_layer, output_layer, hidden_weights, output_weights, bias, iterations):
    for _ in range(iterations):
        a2, a3, hidden_weights, output_weights = feedForwardProp(
            input_layer, output_layer, hidden_weights, output_weights, bias)

        error_a3 = output_layer - a3 #에러 구하는 식 -> 지금 나의 값과 변경 값의 차이를 구해줌
        error_a2 = np.dot(error_a3, output_weights[0:2, :].T) * \
            sigmoid(np.dot(input_layer, hidden_weights))

        delta_a3 = error_a3 * sigmoid_gradient(a3)
        delta_a2 = error_a2 * sigmoid_gradient(a2[:, 0:2])

        # Update weights
        output_weights += np.dot(a2.T, delta_a3)
        hidden_weights += np.dot(input_layer.T, delta_a2)

    return a3


# Data
input_layer = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]) #list를 array로 변경-> 포인터가 없어져서 훨씬 관리가 편해짐
output_layer = np.array([[0, 1, 1, 0]]).T #T=Transpose -> 행과 열 위치 바꿈 -> 배열 변경 ->행렬을 이용해 병렬로 계산하기 위함


# Randomly initialising weights
np.random.seed(1)
hidden_weights = np.random.random((3, 2)) #학습 전 랜덤하게 초기값 생성
output_weights = np.random.random((3, 1))

# Number of iterations
iterations = 10000

# Bias term
bias = np.ones((1, 4))

print(backPropogation(input_layer, output_layer, hidden_weights, output_weights, bias, iterations))

