from Value import Value
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    # 一个神经元，输入是x，输出是tanh(wx+b)
    def __init__(self,input_size):
        self.W = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        out = sum([wi*xi for wi,xi in zip(self.W,x)])+self.b
        out = out.tanh()
        return out
    
    def parameters(self):
        return self.W+[self.b]

class Leaner(Module):
    
    def __init__(self,input_size,output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self,x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class mlp(Module):

    def __init__(self,input_size,output_sizes):
        # input_size:输入维度,output_sizes:每个隐层的输出维度
        sz=[input_size]+output_sizes
        self.layers=[Leaner(sz[i],sz[i+1]) for i in range(len(output_sizes))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]