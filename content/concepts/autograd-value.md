---
id: autograd-value
title: A number that remembers
chapter: machinery
order: 1
difficulty: 2
summary: "Value is a float with two extra fields: a gradient slot, and a list of the Values that produced it. Every arithmetic operator builds one node linking to its inputs, and that link is the entire basis of automatic differentiation."
math: |
  \text{Value}(d,\; c,\; g)\quad\text{where } c = (\text{parents}),\;\; g = (\partial d / \partial c_i)
shapes:
  - { name: "Value.data", shape: "float" }
  - { name: "Value._children", shape: "tuple[Value, ...]" }
  - { name: "Value._local_grads", shape: "tuple[float, ...]" }
anchors:
  python: { selector: "def:Value" }
  c: { selector: "def:backward_all offset:[0, 22]" }
related: [autograd-backward, params-init, linear]
prereqs: []
---

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

Four fields. Read them twice, because this is the hinge of the whole file:

- `data` — the number. This is the only thing the forward pass cares about.
- `grad` — the derivative of the final loss with respect to `data`. Zero until
  `backward()` runs.
- `_children` — the Values that were combined to produce this one.
- `_local_grads` — how much each of those children contributed, *locally*.

## The idea

Ordinary Python `+` throws away its history. `1.0 + 2.0` gives you `3.0` and
nothing about how to get back to `1.0`. That is fine for arithmetic and fatal for
a neural network, which needs `d(loss)/d(weight)` for all 4,192 weights.

So `__add__` returns a new `Value` that *remembers* its operands:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

The local gradients are `(1, 1)` because `d(a+b)/da = d(a+b)/db = 1`. The
constructor is handed `data`, the parent tuple, and the local-derivative tuple —
and that is the whole contract. After the forward pass, every number in the
program is sitting inside a little graph node with its edges intact.

## Every operator is the same three lines

```python
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
def log(self):          return Value(math.log(self.data), (self,), (1/self.data,))
def exp(self):          return Value(math.exp(self.data), (self,), (math.exp(self.data),))
def relu(self):         return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

Each one is: the forward value, who the parent is, and the derivative at this
point. `relu` is the clearest — its derivative is 1 above zero, 0 below, and
ambiguous exactly at zero, where this file silently picks 0.

`__mul__` is the one to understand properly:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

The local gradients are *the other operand's value*. That is the product rule:
`d(ab)/da = b`, `d(ab)/db = a`. Nothing memorises the rule table; the local
gradient just happens to be the sibling's data.

## The reverse operators, for free

```python
def __neg__(self): return self * -1
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __truediv__(self, other): return self * other**-1
```

Every one is defined in terms of something already working. `a / b` is
`a * b**-1`, and `b**-1` is a `__pow__` node whose local gradient is
`-1 * b**-2`. Four lines of definition, and the graph never grows a special case.
The `r`-prefixed variants exist purely so that `3 - value` and `3 / value` work
at all; without them Python would try `int.__sub__(Value)` and fail.

## `__slots__`

```python
__slots__ = ('data', 'grad', '_children', '_local_grads')
```

Every single `Value` in this program is one of these. During the backward pass
the graph holds, per document, on the order of tens of thousands of them. Cutting
the per-object `__dict__` — and the memory that goes with it — is the difference
between a file that trains in a minute and one that does not. It is a one-line
change to a class you are reading as a teaching example, and it is load-bearing.

## The C port has no `Value` at all

Look at the C tab. There is no autograd. The `Value` class and its 40 lines of
operators are simply absent, replaced by `static float` gradient arrays and a
hand-written `backward_all`.

This is the most important thing to notice when reading across languages, so it
is worth being blunt about: **the C port does not compute the same derivatives
automatically, it computes them because somebody typed them out.** Every
`dL/d(...)` in that file is a human remembering a chain rule, and the reader has
no way to check that the human was right by looking at the code alone — which is
precisely the class of bug autograd exists to make impossible.

Two consequences worth internalising:

- Autograd trades memory and speed for *correctness by construction*. That is a
  good trade at 4,192 parameters and a much worse one at 175 million, which is
  why the scaled C config hand-writes its backward pass too.
- It also means the Python file is not "the C file, but slower". It is a
  different program that computes the same function. The only thing they have in
  common is the answer.

## What this does *not* do

There is no operator for matrix multiply, no broadcasting, no batch dimension,
and no GPU. There is also no `addcmul`, no `pow` with an exponent that is itself
a `Value`, and no second-order derivative. All of that is fine — the point of
this file is that the *idea* fits in a class this small, and every production
autograd engine is this class plus bookkeeping.

:::callout "Try it"
Change `relu`'s local gradient to `1.0` and rerun. The loss still falls, because
you have only changed the gradient in the region where activations are positive
for the names in this dataset. That is a useful reminder that a gradient bug and
an architecture bug look identical from the loss curve, and that the only defense
is testing the derivative directly — which is exactly what this repository's
finite-difference tests do for the C port.
:::
