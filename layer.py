from matrix import Matrix
import math
# from neuralNetwork import ActivationFunc

class Layer:
    def __init__(self, num_nodes, prev_nodes, activation_func, preset_Activation=True, nodes=None, weights=None, biases=None):
        self.num_nodes = num_nodes
        self.nodes = nodes
        if (preset_Activation):
            self.activation_func = ActivationFunc.from_preset(activation_func)
            self.activation_pointer = activation_func
        else:
            self.activation_func = activation_func
            self.activation_pointer = 'unable to save'


        if weights == None:
            self.weights = Matrix(num_nodes, prev_nodes)
            self.weights.randomize()
        else:
            self.weights = Matrix.copy(weights)
        
        if biases == None:
            self.biases = Matrix(num_nodes, 1)
            self.biases.randomize()
        else:
            self.biases = Matrix.copy(biases)
    
    def get_dict(self, nodes=False):
        layer_dict = {
            'num_nodes': self.num_nodes,
            'activation_func': self.activation_pointer,
            'nodes': None,
            'weights': self.weights.__dict__,
            'biases': self.biases.__dict__,
        }
        if (nodes): 
            layer_dict['nodes'] = self.nodes.__dict__
        return layer_dict
    
    @classmethod
    def from_dict(cls, layer_dict, prev_nodes):
        if layer_dict['nodes'] == None:
            nodes = None
        else:
            nodes = Matrix.from_array2d(layer_dict['nodes']['arr'])
        weights = Matrix.from_array2d(layer_dict['weights']['arr'])
        biases = Matrix.from_array2d(layer_dict['biases']['arr'])

        layer = cls(layer_dict['num_nodes'], prev_nodes, layer_dict['activation_func'], nodes=nodes, weights=weights, biases=biases)

        return layer        

class ActivationFunc:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc
    
    @classmethod
    def from_preset(cls, func):
        if func == "sigmoid":
            return cls(cls.sigmoid, cls.dsig_applied)
        else: 
            return None
            
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    @staticmethod
    def dsigmoid(x):
        return ActivationFunc.sigmoid(x)*(1 - ActivationFunc.sigmoid(x))
    @staticmethod
    def dsig_applied(x):
        return x *(1 - x)