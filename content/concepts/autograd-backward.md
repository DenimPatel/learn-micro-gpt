---
id: autograd-backward
title: Walking the graph backwards
chapter: machinery
order: 2
difficulty: 2
summary: "backward() builds a topological order of the computation graph, then walks it in reverse accumulating child.grad += local_grad * parent.grad. One method, fourteen lines, and the chain rule is a for loop."
math: |
  \frac{\partial L}{\partial c_i} \mathrel{+}= \frac{\partial v}{\partial c_i}\cdot \frac{\partial L}{\partial v}, \qquad \forall\, i \in \mathrm{children}(v)
shapes:
  - { name: topo, shape: "list[Value], parents before children" }
  - { name: grads, shape: "one float per Value" }
anchors:
  python: { selector: "def:Value.backward" }
  c: { selector: "def:backward_all" }
related: [autograd-value, adam, cross-entropy-loss]
prereqs: [autograd-value]
---

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

That is the entire backward pass. Two phases.

## Phase 1: sort the graph

`build_topo` is a depth-first walk that appends each node *after* its children,
so `topo` is ordered with parents first. Reversing it therefore gives nodes
children-first, which is the only order in which the chain rule is computable:
you cannot know `d(loss)/d(x)` until every path from the loss down to `x` has
been accounted for.

`visited` is a `set` of `Value` objects, so it relies on `__eq__`/`__hash__`.
`Value` defines neither, which means every `Value` hashes by identity — correct,
and only correct as long as nobody ever defines `__eq__`. Combined with
`__slots__`, that also means the `set` holds references to the whole graph, so
the graph cannot be collected early.

## Phase 2: accumulate

```python
self.grad = 1
```

The loss is the root, and `d(loss)/d(loss) = 1`. Everything else follows from
this one seed.

```python
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

Read that as the chain rule, literally: the derivative of the loss with respect
to `child` is the local derivative of `v` with respect to `child`, times the
derivative of the loss with respect to `v`. Summed over every parent, because a
node can be used in many places and the gradients accumulate.

The `+=` is the load-bearing character. `child.grad = ...` would be wrong the
moment a `Value` is reused — and it is reused constantly, because
`state_dict` rows are shared by every document, every position, and every
head. One weight, hundreds of contributions, summed.

## The result

After this, every `Value` in the graph has a `.grad`. The training loop then does
`loss.backward()` and immediately walks `params` applying Adam. Nothing else
touches the graph. It is rebuilt from scratch on the next step, because
`gpt()` constructs fresh `Value`s every call and nothing is cached.

:::callout "This is why the reference is slow"
The graph is built and torn down 1,000 times, for one document at a time, with
pure-Python object allocation for every one of tens of thousands of nodes per
step. The C port gets its speed from the same arithmetic laid out in typed
arrays, and roughly 100x from batching positions instead of stepping through
them one at a time. The algorithm is identical; the bookkeeping is not.
:::

## Reuse, and the one thing to be careful about

Because the graph is rebuilt every step, `p.grad` is reset to `0` at the bottom
of the Adam update. That line looks redundant — `backward()` just accumulated
into it — and deleting it will appear to work for a while before producing
gradients that are the sum of everything ever computed. The comment says
`p.grad = 0` for a reason.

:::callout "The debugging habit this trains"
`loss.backward()` mutates the entire graph. If you want to backward twice from
the same loss, you cannot: the gradients are already accumulated and there is no
zeroing step. `build_topo` is also recursive, so a graph deeper than Python's
recursion limit (~1,000) would blow the stack. At `n_layer = 1` that is not
close, but it is the wall you hit first when you try to make this model deeper.
That is a real limitation, and it is why production engines are iterative.
:::
