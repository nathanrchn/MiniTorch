from .vector import Vector
from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self) -> None:
        super().__setattr__("_parameters", dict())

    def __setattr__(self, name: str, value: object) -> None:
        if issubclass(type(value), Vector):
            self._parameters[name] = value
        elif issubclass(type(value), list):
            for i, v in enumerate(value):
                if issubclass(type(v), Vector):
                    self._parameters[f"{name}.{i}"] = v
        elif issubclass(type(value), Module):
            for k, v in value._parameters.items():
                self._parameters[f"{name}.{k}"] = v
        
        super().__setattr__(name, value)

    def __call__(self, x: Vector) -> Vector:
        return self.forward(x)

    def parameters(self) -> list[Vector]:
        return list(self._parameters.values())
    
    @abstractmethod
    def forward(self, x: Vector) -> Vector:
        pass

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.weight: list[Vector] = [Vector.randn(in_features, requires_grad=True) for _ in range(out_features)]

        if self.bias:
            self.bias_vector: Vector = Vector.randn(out_features, requires_grad=True)

    def forward(self, x: Vector) -> Vector:
        out = Vector([0.0] * len(self.weight), requires_grad=True)
        for i, w in enumerate(self.weight):
            out.data[i] = x.dot(w).data[0]
        if self.bias:
            out += self.bias_vector
        return out