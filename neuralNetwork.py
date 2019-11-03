# from matplotlib import pyplot as plt
import math
import random
import pickle
import json

from matrix import Matrix
from layer import Layer
from layer import ActivationFunc


class NeuralNetwork:
    # sigmoid = ActivationFunc(
    #     ActivationFunc.sigmoid,
    #     ActivationFunc.dsig_applied
    # )

    def __init__(self, input_nodes=None, layers=None, lr=None, epochs=0):
        self.input_nodes = input_nodes
        if (layers == None):
            self.layers = []
        else:
            self.layers = layers
        self.lr = lr
        self.epochs = epochs

    def set_input_nodes(self, input_nodes):
        self.input_nodes = input_nodes

    def set_lr(self, lr):
        self.lr = lr
    
    def add_layer(self, nodes, activation_func, preset_Activation=True, weights=None, biases=None):
        if len(self.layers) == 0:
            layer = Layer(nodes, self.input_nodes, activation_func, preset_Activation, weights, biases)
        else:
            layer = Layer(nodes, self.layers[-1].num_nodes, activation_func,preset_Activation, weights, biases)
        self.layers.append(layer)
    
    def feed_forward(self, inputs):
        self.layers[0].nodes = Matrix.mat_mul(self.layers[0].weights, inputs) 
        self.layers[0].nodes.add(self.layers[0].biases)
        self.layers[0].nodes.map(self.layers[0].activation_func.func)

        for i in range(1, len(self.layers)):
            self.layers[i].nodes = Matrix.mat_mul(self.layers[i].weights, self.layers[i-1].nodes)
            self.layers[i].nodes.add(self.layers[i].biases)
            self.layers[i].nodes.map(self.layers[i].activation_func.func)
        
        return self.layers[-1]
    
    def predict(self, input_arr):
        return Matrix.to_array(self.feed_forward(Matrix.from_array(input_arr)).nodes)
    
    def back_propogate(self, input_arr, targets_arr):  
        inputs = Matrix.from_array(input_arr)
        targets = Matrix.from_array(targets_arr)
        outputs = self.feed_forward(inputs)

        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1:
                error = Matrix.s_sub(targets, outputs.nodes)
            else:
                error = Matrix.mat_mul(self.layers[i+1].weights.tr, error)

            gradients = Matrix.s_map(self.layers[i].nodes, self.layers[i].activation_func.dfunc)
            gradients.mult(error)
            gradients.mult(self.lr) 
            
            if i == 0:
                delta_weights = Matrix.mat_mul(gradients, inputs.tr)
            else:
                delta_weights = Matrix.mat_mul(gradients, self.layers[i-1].nodes.tr)

            self.layers[i].weights.add(delta_weights)
            self.layers[i].biases.add(gradients)
    
    def train(self, x_train, y_train, epochs):
        print('\nTraining:')
        for i in range(epochs):
            self.epochs += 1
            print(' epoch {}...'.format(self.epochs))
            xs, ys = Matrix.shuffle(x_train, y_train)
            for j in range(len(xs)):
                self.back_propogate(xs[j], ys[j])
            print(' Done epoch {}!'.format(self.epochs))
        print('Done Training')
    
    def test(self, x_test, y_test):
        print('\nTesting...')
        acc = 0
        xs, ys = Matrix.shuffle(x_test, y_test)
        for i in range(len(xs)):
            guess = self.predict(xs[i])
            if Matrix.arg_max(guess) == Matrix.arg_max(ys[i]):
                acc += 1
        acc /= len(xs)
        print('Done testing! accuracy: {:.5f}%'.format(acc*100))
        return acc

    def get_dict(self, nodes=False):
        model_dict = {
            'input_nodes': self.input_nodes,
            'layers': [],
            'lr': self.lr,
            'epochs': self.epochs
        }
        for layer in self.layers:
            model_dict['layers'].append(layer.get_dict(nodes))
        return model_dict

    def save(self, dir, format=False, nodes=False):
        with open(dir, 'w') as f:
            if format:
                json.dump(self.get_dict(nodes), f, indent=2)
            else:
                json.dump(self.get_dict(nodes), f, )
        
    @classmethod
    def load(cls, dir):
        with open(dir, 'r') as f:
            data = json.load(f)
        layers = []
        for layer in data['layers']:
            if len(layers) == 0:
                layers.append(Layer.from_dict(layer, data['input_nodes']))
            else:
                layers.append(Layer.from_dict(layer, data['layers'][-1]['num_nodes']))

        model = cls(input_nodes=data['input_nodes'], layers=layers, lr=data['lr'], epochs=data['epochs'])
        return model

            



        

def get_label(input_arr):
    if input_arr[0] + input_arr[1] == 1:
        return [0, 1]
    else:
        return [1, 0]

def main():
    nn = NeuralNetwork()
    nn.set_input_nodes(2)
    nn.add_layer(4, 'sigmoid')
    nn.add_layer(4, 'sigmoid')
    nn.add_layer(2, 'sigmoid')
    nn.set_lr(0.1)


    x_train = []
    y_train = []
    for i in range(1000):
        x_train.append([])
        x_train[i].append(round(random.random()))
        x_train[i].append(round(random.random()))
        y_train.append(get_label(x_train[i]))
    
    print(len(x_train), len(x_train[0]))
    nn.train(x_train, y_train, 3)
    
    x_test = []
    y_test = []
    for i in range(1000):
        x_test.append([])
        x_test[i].append(round(random.random()))
        x_test[i].append(round(random.random()))
        y_test.append(get_label(x_test[i]))
    
    nn.test(x_test, y_test)

    print()
    print('0 0 = {}'.format(Matrix.arg_max(nn.predict([0,0]))))
    print('0 1 = {}'.format(Matrix.arg_max(nn.predict([0,1]))))
    print('1 0 = {}'.format(Matrix.arg_max(nn.predict([1,0]))))
    print('1 1 = {}'.format(Matrix.arg_max(nn.predict([1,1]))))
    
    # nn.save('XOR.pickle')
    # model = NeuralNetwork.load('XOR.pickle')
    nn.save('model.json', format=True)
    model = NeuralNetwork.load('model.json')
    model.train(x_train, y_train, 3)

    print()
    print('0 0 = {}'.format(Matrix.arg_max(model.predict([0,0]))))
    print('0 1 = {}'.format(Matrix.arg_max(model.predict([0,1]))))
    print('1 0 = {}'.format(Matrix.arg_max(model.predict([1,0]))))
    print('1 1 = {}'.format(Matrix.arg_max(model.predict([1,1]))))

    # print(nn.get_dict())
    # print(nn.layers[0].get_dict())

    # json_nn = json.dumps(nn.get_dict(), indent=2)
    # print(json_nn)

    # print(model.__dict__)
    # print(model.layers[1].__dict__)
    # print(model.layers[1].weights.__dict__)

if __name__ == "__main__":
    main()    

    
 

