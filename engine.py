import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        assert isinstance(other, (Value, int, float)), "not support value type"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            other += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        assert isinstance(other, (Value, int, float)), "not support value type"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "not support value type"
        out = Value(self.data**other, (self,), f"^{other}")

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        v = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(v, (self,), "tanh")

        def _backward():
            self.grad += (1 - v**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        v = x if x > 0 else 0
        out = Value(v, (self,), "relu")

        def _backward():
            self.grad += out.grad if x > 0 else 0
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

