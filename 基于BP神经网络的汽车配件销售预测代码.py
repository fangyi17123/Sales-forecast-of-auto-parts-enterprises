# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:30:10 2021

@author: 92102
"""
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from matplotlib import pyplot as plt
import os
import sys

train_batch_size = 10#训练批次
test_batch_size = 100#测试批次
num_iterations = 1000
lr = 0.002#学习率
weight_decay = 0.02#权重更新
num_neuron=12#隐藏层节点数
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

class DataHander:
    #数据处理
    def __init__(self,batch_size):
        self.data_sample = 0
        self.data_label = 0
        self.output_sample = 0
        self.output_label = 0
        self.point = 0  # 用于记住下一次pull数据的地方;
        self.batch_size = batch_size

    def get_data(self, sample, label): 
        self.data_sample = sample
        self.data_label = label

    def shuffle(self):  
        random_sequence = random.sample(range(self.data_sample.shape[0]), self.data_sample.shape[0])
        self.data_sample = self.data_sample[random_sequence]
        self.data_label = self.data_label[random_sequence]

    def pull_data(self):  
        start = self.point
        end = start + self.batch_size
        output_index = np.arange(start, end)
        if end > self.data_sample.shape[0]:
            end = end - self.data_sample.shape[0]
            output_index = np.append(np.arange(start, self.data_sample.shape[0]), np.arange(0, end))
        self.output_sample = self.data_sample[output_index]
        self.output_label = self.data_label[output_index]
        self.point = end % self.data_sample.shape[0]
        
class Optimizer():
    #优化策略
    def __init__(self, lr = 0.01, momentum = 0.9, iteration = -1, gamma=0.0005, power=0.75):#初始化
        self.lr = lr
        self.momentum = momentum
        self.iteration = iteration
        self.gamma = gamma
        self.power = power
    
    def fixed(self):
        return self.lr

    def anneling(self):
        if self.iteration == -1:
            assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
        self.lr = self.lr * np.power((1 + self.gamma * self.iteration), -self.power)
        return self.lr

    def batch_gradient_descent_fixed(self, weights, grad_weights, previous_direction):
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def batch_gradient_descent_anneling(self, weights, grad_weights, previous_direction):
        self.lr = self.anneling()
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def update_iteration(self, iteration):
        self.iteration = iteration

class Initializer:
    def xavier(self, num_neuron_inputs, num_neuron_outputs):
        temp1 = np.sqrt(6) / np.sqrt(num_neuron_inputs + num_neuron_outputs + 1)
        weights = stats.uniform.rvs(-temp1, 2 * temp1, (num_neuron_inputs, num_neuron_outputs))
        return weights

class FullyConnecte():
    #全连接层
    def __init__(self, num_neuron_inputs, num_neuron_outputs, batch_size=10,weights_decay=0.001):
        self.num_neuron_inputs = num_neuron_inputs
        self.num_neuron_outputs = num_neuron_outputs
        self.inputs = np.zeros((batch_size, num_neuron_inputs))
        self.outputs = np.zeros((batch_size, num_neuron_outputs))
        self.weights = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias = np.zeros(num_neuron_outputs)
        self.weights_previous_direction = np.zeros((num_neuron_inputs, num_neuron_outputs))
        self.bias_previous_direction = np.zeros(num_neuron_outputs)
        self.grad_weights = np.zeros((batch_size, num_neuron_inputs, num_neuron_outputs))
        self.grad_bias = np.zeros((batch_size, num_neuron_outputs))
        self.grad_inputs = np.zeros((batch_size, num_neuron_inputs))
        self.grad_outputs = np.zeros((batch_size, num_neuron_outputs))
        self.batch_size = batch_size
        self.weights_decay = weights_decay

    def initialize_weights(self, initializer):
        self.weights = initializer(self.num_neuron_inputs, self.num_neuron_outputs)


    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.inputs.dot(self.weights)+ np.tile(self.bias, (self.batch_size, 1))

    # 在反向传播过程中,用于获取输入;
    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        for i in np.arange(self.batch_size):
            self.grad_weights[i, :] = np.tile(self.inputs[i, :], (1, 1)).T.dot(np.tile(self.grad_outputs[i, :], (1, 1))) + \
            self.weights * self.weights_decay
        self.grad_bias = self.grad_outputs
        self.grad_inputs = self.grad_outputs.dot(self.weights.T)

    def update(self, optimizer):
        grad_weights_average = np.mean(self.grad_weights, 0)
        grad_bias_average = np.mean(self.grad_bias, 0)
        (self.weights, self.weights_previous_direction) = optimizer(self.weights, grad_weights_average,self.weights_previous_direction)
        (self.bias, self.bias_previous_direction) = optimizer(self.bias,grad_bias_average, self.bias_previous_direction)

    def update_batch_size(self,batch_size):
        self.batch_size = batch_size

    

class Activation():
    #激活层
    def __init__(self, activation_function_name):
        if activation_function_name == 'sigmoid':
            self.activation_function = self.sigmoid
            self.der_activation_function = self.der_sigmoid
        elif activation_function_name == 'tanh':
            self.activation_function = self.tanh
            self.der_activation_function = self.der_tanh
        elif activation_function_name == 'relu':
            self.activation_function = self.relu
            self.der_activation_function = self.der_relu
        elif activation_function_name == 'linear':
            self.activation_function = self.identity
            self.der_activation_function = self.der_identity
        else:
            print('wrong activation function')
        self.inputs = 0
        self.outputs = 0
        self.grad_inputs = 0
        self.grad_outputs = 0

    def get_inputs_for_forward(self, inputs):
        self.inputs = inputs

    def forward(self):
        self.outputs = self.activation_function(self.inputs)

    def get_inputs_for_backward(self, grad_outputs):
        self.grad_outputs = grad_outputs

    def backward(self):
        self.grad_inputs = self.grad_outputs * self.der_activation_function(self.inputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def der_tanh(self, x):
        return 1 - self.tanh(x) * self.tanh(x)

    def relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_zero = (x > temp)
        return x * if_bigger_zero

    def der_relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_equal_zero = (x >= temp)  # 在零处的导数设为1
        return if_bigger_equal_zero * np.ones_like(x)

    def identity(self, x):
        return x

    def der_identity(self, x):
        return x



class Loss():
    #损失函数
    def __init__(self, loss_function_name):
        self.inputs = 0
        self.loss = 0
        self.grad_inputs = 0
        if loss_function_name == 'SoftmaxLogloss':
            self.loss_function = self.softmax_logloss
            self.der_loss_function = self.der_softmax_logloss
        elif loss_function_name == 'LeastSquareLoss':
            self.loss_function = self.least_square_loss
            self.der_loss_function = self.der_least_square_loss
        else:
            print("wrong loss function")
    def get_label_for_loss(self, label):
        self.label = label

    def get_inputs_for_loss(self, inputs):
        self.inputs = inputs

    def compute_loss(self):
        self.loss = self.loss_function(self.inputs, self.label)

    def compute_gradient(self):
        self.grad_inputs = self.der_loss_function(self.inputs, self.label)
        
    def softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        probability = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
        temp3 = np.argmax(label, 1)  
        temp4 = [probability[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp3)]
        loss = -1 * np.mean(np.log(temp4))
        return loss

    def der_softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        temp2 = np.sum(temp1, 1)
        probability = temp1 / (np.tile(temp2, (inputs.shape[1], 1))).T
        gradient = probability - label
        return gradient

    def sigmoid_logloss(self, inputs, label):
        probability = np.array([(1.0 / (1 + np.exp(-i))) for i in inputs])
        loss = - np.sum(np.dot(label.T,np.log(probability)+ np.dot((1-label).T,np.log(1-probability)))) / ( len(label))
        return loss

    def der_sigmoid_logloss(self, inputs, label):
        probability = np.array([(1.0 / (1 + np.exp(-i))) for i in inputs])
        gradient = label - probability
        return gradient

    def least_square_loss(self, predict, label):
        tmp1 = np.sum(np.square(label - predict), 1)
        loss = np.mean(tmp1)
        return loss

    def der_least_square_loss(self, predict, label):
        gradient = predict - label
        return gradient


class MetricCalculator():
    #训练测试指标
    def __init__(self, label, predict):
        self.label = label
        self.predict = predict
        assert len(label)==len(predict), "length of label and predict must be equal"
        self.mse = None
        self.rmse = None
        self.mae = None
        self.auc = None

    def get_mse(self):
        self.mse = np.mean(np.sum(np.square(self.label - self.predict),1))

    def get_rmse(self):
        self.rmse = np.sqrt(np.mean(np.sum(np.square(self.label - self.predict), 1)))

    def get_mae(self):
        self.mae = np.mean(np.sum(np.abs(self.label - self.predict),1))

    def get_auc(self):
        prob = self.predict.reshape(-1).tolist()
        label = self.label.reshape(-1).tolist()
        f = list(zip(prob, label))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(label)):
            if (label[i] == 1):
                posNum += 1
            else:
                negNum += 1
        self.auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)

    def print_metrics(self):
        if(self.mse): print("mse: ",self.mse)
        if(self.rmse): print("rmse: ",self.rmse)
        if(self.mae): print("mae: ",self.mae)
        if(self.auc): print("auc: ",self.auc)



class BPNet():
    #BP网络
    def __init__(self, optimizer = Optimizer.batch_gradient_descent_fixed, initializer = Initializer.xavier, \
                 batch_size=train_batch_size, num_neuron=num_neuron,weights_decay=0.001):
        self.optimizer = optimizer
        self.initializer = initializer
        self.batch_size = batch_size
        self.weights_decay = weights_decay
        self.fc1 = FullyConnecte(6,num_neuron,self.batch_size, self.weights_decay)
        self.ac1 = Activation('relu')
        self.fc2 = FullyConnecte(num_neuron,1,self.batch_size, self.weights_decay)
        self.loss = Loss("LeastSquareLoss")

    def forward_train(self,input_data, input_label):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()

        self.loss.get_inputs_for_loss(self.fc2.outputs)
        self.loss.get_label_for_loss(input_label)
        self.loss.compute_loss()

    def backward_train(self):
        self.loss.compute_gradient()
        self.fc2.get_inputs_for_backward(self.loss.grad_inputs)
        self.fc2.backward()
        self.ac1.get_inputs_for_backward(self.fc2.grad_inputs)
        self.ac1.backward()
        self.fc1.get_inputs_for_backward(self.ac1.grad_inputs)
        self.fc1.backward()

    def predict(self,input_data):
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()

        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        return self.fc2.outputs

    def eval(self,input_data, input_label):
        self.fc1.update_batch_size(input_data.shape[0])
        self.fc1.get_inputs_for_forward(input_data)
        self.fc1.forward()
        self.ac1.get_inputs_for_forward(self.fc1.outputs)
        self.ac1.forward()
        self.fc2.update_batch_size(input_data.shape[0])
        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        #print("predict: \n ",self.fc2.outputs[:10])
        #print("label: \n", input_label[:10])
        metric = MetricCalculator(label=input_label, predict=self.fc2.outputs)
        metric.get_mae()
        metric.get_mse()
        metric.get_rmse()
        metric.print_metrics()

    def update(self):
        self.fc1.update(self.optimizer)
        self.fc2.update(self.optimizer)

    def initial(self):
        self.fc1.initialize_weights(self.initializer)
        self.fc2.initialize_weights(self.initializer)





if __name__ == "__main__":
    #主函数入口
    sales_forecast_data = pd.read_excel("历史数据.xlsx")
    print("data_shape:", sales_forecast_data.shape)

    data_sample = sales_forecast_data.iloc[:, :-1].values
    data_label = sales_forecast_data.iloc[:, -1].values.reshape(-1,1)

    mean = data_sample.mean(axis=0)
    std = data_sample.std(axis=0)
    data_sample = (data_sample-mean)/std

    data_length = data_label.shape[0]
    train_data_length = int(data_length * 0.8)
    print("train_label_length:",train_data_length)
    
    data_sample_train, data_sample_test = data_sample[:train_data_length], data_sample[train_data_length:]
    data_label_train, data_label_test = data_label[:train_data_length], data_label[train_data_length:]
    


    data_handler = DataHander(train_batch_size)
    opt = Optimizer(lr = lr,momentum = 0.9,iteration = 0,gamma = 0.0005,power = 0.75)
    initializer = Initializer()
    data_handler.get_data(sample=data_sample_train,label=data_label_train)
    data_handler.shuffle()
    
    bpnet = BPNet(optimizer = opt.batch_gradient_descent_anneling, initializer = initializer.xavier, batch_size = train_batch_size, \
                weights_decay = weight_decay)
    bpnet.initial()
    
    train_error = []
    max_loss = math.inf
    early_stopping_iter = 35
    early_stopping_mark = 0
    
    for i in range(num_iterations):
        opt.update_iteration(i)
        data_handler.pull_data()
        bpnet.forward_train(data_handler.output_sample,data_handler.output_label)
        bpnet.backward_train()
        bpnet.update()
        train_error.append(bpnet.loss.loss)
        if max_loss >  bpnet.loss.loss:
            early_stopping_mark = 0
            max_loss = bpnet.loss.loss
        if early_stopping_mark > early_stopping_iter:
            break
        early_stopping_mark += 1
    
    plt.plot(train_error)
    plt.show()
    
    #测试
    bpnet.eval(data_sample_test,data_label_test)


