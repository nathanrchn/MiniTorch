from math import log, exp
from random import uniform

class Vector:
    def __init__(self, data: list, requires_grad: bool = False, _children: tuple =()) -> None:
        self.data: list = data
        self.grad: Vector = None
        self._prev: set = set(_children)
        self.requires_grad: bool = requires_grad
        self._backward_fn: function = lambda: None

        if self.requires_grad:
            self.grad = Vector([0.0] * len(data), requires_grad=False)

    @classmethod
    def randn(cls, length: int, requires_grad: bool = False) -> "Vector":
        data: list = [uniform(-1, 1) for _ in range(length)]
        return cls(data, requires_grad)
    
    def backward(self, grad: "Vector" = None) -> None:
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if grad:
            grad.requires_grad = False
            self.grad = grad
        else:
            self.grad = Vector([1.0] * len(self.data), requires_grad=False)
        for v in reversed(topo):
            v._backward_fn()

    def dot(self, other: "Vector") -> "Vector":
        assert isinstance(other, Vector), "Vector dot product only supports Vector"
        assert len(self) == len(other), "Vector size mismatch"
        
        children: tuple[Vector] = ()
        if self.requires_grad:
            children += (self,)
        if other.requires_grad:
            children += (other,)
        requires_grad = self.requires_grad or other.requires_grad
        out = Vector([sum(x * y for x, y in zip(self.data, other.data))], requires_grad, children)
        
        if requires_grad:
            def dotbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + y * out.grad.data[0] for x, y in zip(self.grad.data, other.data)]
                if other.requires_grad:
                    other.grad.data = [x + y * out.grad.data[0] for x, y in zip(other.grad.data, self.data)]
            out._backward_fn = dotbackward_fn

        return out
    
    def exp(self) -> "Vector":
        out = Vector([exp(x) for x in self.data], self.requires_grad, (self,))

        if self.requires_grad:
            def expbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + y * z for x, y, z in zip(self.grad.data, out.data, out.grad.data)]
            out._backward_fn = expbackward_fn
    
        return out
    
    def log(self) -> "Vector":
        out = Vector([log(x) for x in self.data], self.requires_grad, (self,))

        if self.requires_grad:
            def logbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + (1 / y) * z for x, y, z in zip(self.grad.data, self.data, out.grad.data)]
            out._backward_fn = logbackward_fn

        return out
    
    def relu(self) -> "Vector":
        out = Vector([max(0.0, x_i) for x_i in self.data], self.requires_grad, (self,))

        if self.requires_grad:
            def relubackward_fn() -> None:
                self.grad.data = [x + (x_i > 0) * y for x, x_i, y in zip(self.grad.data, self.data, out.grad.data)]
            out._backward_fn = relubackward_fn

        return out
    
    def sigmoid(self) -> "Vector":
        out = Vector([(1 / (1 + exp(-x))) for x in self.data], self.requires_grad, (self,))

        if self.requires_grad:
            def sigmoidbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + y * (1 - y) * z for x, y, z in zip(self.grad.data, out.data, out.grad.data)]
            out._backward_fn = sigmoidbackward_fn

        return out
    
    def softmax(self) -> "Vector":
        m = max(self.data)
        exps = [exp(x - m) for x in self.data]
        divisor = sum(exps)
        out = Vector([exp / divisor for exp in exps], self.requires_grad, (self,))

        if self.requires_grad:
            def softmaxbackward_fn() -> None:
                if self.requires_grad:
                    jacobian = []
                    for i in range(len(self.data)):
                        row = []
                        for j in range(len(self.data)):
                            if i == j:
                                row.append(out.data[i] * (1 - out.data[i]))
                            else:
                                row.append(-out.data[i] * out.data[j])
                        jacobian.append(row)
                    
                    self.grad.data = [x + sum(y * z for y, z in zip(row, out.grad.data)) for x, row in zip(self.grad.data, jacobian)]
            out._backward_fn = softmaxbackward_fn

        return out
    
    def sum(self) -> "Vector":
        out = Vector([sum(self.data)], self.requires_grad, (self,))

        if self.requires_grad:
            def sumbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + out.grad.data[0] for x in self.grad.data]
            out._backward_fn = sumbackward_fn

        return out
    
    def tanh(self):
        out = Vector([(exp(x) - exp(-x))/(exp(x) + exp(-x)) for x in self.data], self.requires_grad, (self,))

        if self.requires_grad:
            def tanhbackward_fn() -> None:
                if self.requires_grad:
                    self.grad.data = [x + (1 - y ** 2) * z for x, y, z in zip(self.grad.data, out.data, out.grad.data)]
            out._backward_fn = tanhbackward_fn

        return out
    
    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad.data = [0.0] * len(self.data)

    def __add__(self, other) -> "Vector":
        children: tuple[Vector] = ()
        if self.requires_grad:
            children += (self,)

        if isinstance(other, (int, float)):
            out = Vector([x + other for x in self.data], self.requires_grad, children)

            if self.requires_grad:
                def addbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + y for x, y in zip(self.grad.data, out.grad.data)]
                out._backward_fn = addbackward_fn
        elif isinstance(other, Vector):
            assert len(self) == len(other), "Vector size mismatch"
            requires_grad = self.requires_grad or other.requires_grad
            if other.requires_grad:
                children += (other,)
            out = Vector([x + y for x, y in zip(self.data, other.data)], requires_grad, children)

            if requires_grad:
                def addbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + y for x, y in zip(self.grad.data, out.grad.data)]
                    if other.requires_grad:
                        other.grad.data = [x + y for x, y in zip(other.grad.data, out.grad.data)]
                out._backward_fn = addbackward_fn
        else:
            raise NotImplementedError
        
        return out
    
    def __mul__(self, other) -> "Vector":
        children: tuple[Vector] = ()
        if self.requires_grad:
            children += (self,)

        if isinstance(other, (int, float)):
            out = Vector([x * other for x in self.data], self.requires_grad, children)

            if self.requires_grad:
                def mulbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + other * y for x, y in zip(self.grad.data, out.grad.data)]
                out._backward_fn = mulbackward_fn
        elif isinstance(other, Vector):
            assert len(self) == len(other), "Vector size mismatch"
            requires_grad = self.requires_grad or other.requires_grad
            if other.requires_grad:
                children += (other,)

            out = Vector([x * y for x, y in zip(self.data, other.data)], requires_grad, children)

            if requires_grad:
                def mulbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + y * z for x, y, z in zip(self.grad.data, other.data, out.grad.data)]
                    if other.requires_grad:
                        other.grad.data = [x + y * z for x, y, z in zip(other.grad.data, self.data, out.grad.data)]
                out._backward_fn = mulbackward_fn
        else:
            raise NotImplementedError
        
        return out
    
    def __pow__(self, other) -> "Vector":
        children: tuple[Vector] = ()
        if self.requires_grad:
            children += (self,)

        if isinstance(other, (int, float)):
            out = Vector([x ** other for x in self.data], self.requires_grad, children)

            if self.requires_grad:
                def powbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + other * (y ** (other - 1)) * z for x, y, z in zip(self.grad.data, self.data, out.grad.data)]
                out._backward_fn = powbackward_fn
        elif isinstance(other, Vector):
            assert len(self) == len(other), "Vector size mismatch"
            requires_grad = self.requires_grad or other.requires_grad
            if other.requires_grad:
                children += (other,)
            out = Vector([x ** y for x, y in zip(self.data, other.data)], requires_grad, children)

            if requires_grad:
                def powbackward_fn() -> None:
                    if self.requires_grad:
                        self.grad.data = [x + y * (z ** (y - 1)) * w for x, y, z, w in zip(self.grad.data, other.data, self.data, out.grad.data)]
                    if other.requires_grad:
                        other.grad.data = [x + log(y) * (y ** z) * w for x, y, z, w in zip(other.grad.data, self.data, other.data, out.grad.data)]
                out._backward_fn = powbackward_fn
        else:
            raise NotImplementedError
        
        return out
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
    
    def __max__(self) -> float:
        return max(self.data)

    def __repr__(self) -> str:
        if self.requires_grad:
            if self._backward_fn.__name__ == "<lambda>":
                return f"vector([{', '.join(f'{x:.4f}' for x in self.data)}], grad={self.grad.data})"
            else:
                return f"vector([{', '.join(f'{x:.4f}' for x in self.data)}], grad={self.grad.data}, backward_fn={self._backward_fn.__name__})"
        else:
            return f"vector([{', '.join(f'{x:.4f}' for x in self.data)}])"