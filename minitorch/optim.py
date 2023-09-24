from .vector import Vector
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, parameters: list[Vector], lr: float = 0.001) -> None:
        self._parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self._parameters:
            p.zero_grad()

    @abstractmethod
    def step(self) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, parameters: list[Vector], lr: float = 0.001) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        for p in self._parameters:
            p.data = [x - self.lr * y for x, y in zip(p.data, p.grad.data)]