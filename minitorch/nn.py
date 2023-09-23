from vector import Vector

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.bias = bias
        self.weight: list[Vector] = [Vector.randn(in_features, requires_grad=True) for _ in range(out_features)]

        if self.bias:
            self.bias_vector: Vector = Vector.randn(out_features, requires_grad=True)

    def forward(self, x: Vector) -> Vector:
        out = Vector([0.0] * len(self.weight), requires_grad=True)
        for i, w in enumerate(self.weight):
            out[i] = x.dot(w)
        if self.bias:
            out += self.bias_vector
        return out
    
    def parameters(self) -> list[Vector]:
        params: list[Vector] = self.weight
        if self.bias:
            params.append(self.bias_vector)
        return params