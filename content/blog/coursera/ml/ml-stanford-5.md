+++
author = "Guocheng Wei"
categories = ["Coursera Notes"]
tags = ["Machine Learning"]
date = "2019-07-03"
description = "Neural Network Cost Function, Backpropagation Algorithm, Unrolling Parameters, Gradient Checking, Random Initialization"
featured = "ml_stanford_5.png"
featuredalt = "ml stanford thumbnail"
featuredpath = "date"
linktitle = ""
title = "Week 5 - Machine Learning"
type = "post"

+++

# Neural Network

---
### Cost Function

<div style="font-size: 70%;">
\begin{gather*}
  \large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2
\end{gather*}
</div>

Some notations:

* L = total number of layers in the network
* $s\_l$ = number of units (not counting bias unit) in layer l
* K = number of output units/classes

Note:

* The double sum simply adds up the logistic regression costs calculated for each cell **in the output layer**
* The triple sum simply adds up the squares of all the individual Θs in the entire network.
* The i in the triple sum does **not** refer to training example i

---
### Backpropagation Algorithm

Our goal is to compute:

$\min\_\Theta J(\Theta)$

In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

$\dfrac{\partial}{\partial \Theta\_{i,j}^{(l)}}J(\Theta)$

In back propagation we're going to compute for every node:

$\delta\_j^{(l)} = \text{"error" of node j in layer l}$

$a\_j^{(l)} = \text{activation node j in layer l}$

For the **last layer**, we can compute the vector of delta values with:

$\delta^{L} = a^{L} - y$

To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

$$
\begin{align}
  & \because &\delta^{(l)} &= ((\Theta^{(l)})^T \delta^{(l+1)})\ .\*\ g'(z^{(l)}) \newline
  & & g'(u) &= g(u) .\* \ ((1− g(u))) \newline \newline
  & \therefore &\delta^{(l)} &= ((\Theta^{(l)})^T \delta^{(l+1)})\ .\* \ a^{(l)}\ .\* \ (1 - a^{(l)})
\end{align}
$$

We can compute our partial derivative terms by multiplying our activation values and our error values for each training example t:

$$
\begin{align}
  \dfrac{\partial J(\Theta)}{\partial \Theta\_{i,j}^{(l)}} = \frac{1}{m}\sum\_{t=1}^m a\_j^{(t)(l)} {\delta}\_i^{(t)(l+1)}
\end{align}
$$

**Note**: This ignores regularization, which we'll deal with later.

#### Algorithm

Given training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$

1. Set $\Delta^{(l)}\_{i,j} := 0 \text{ for all (l)}$

2. For training example t = 1 to m, Set $a^{(1)} := x^{(i)}$

3. Perform forward propagation to compute $a^{(l)} \text{ for all l = 2, 3, ..., l}$

4. $\delta^{L} = a^{L} - y$

5. Compute $\delta^{(L - 1)},\ \delta^{(L - 2)},\ ..., \delta^{(2)}$ using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .\*\ a^{(l)}\ .\*\ (1 - a^{(l)})$

6. $\Delta^{(l)}\_{i,j} := \Delta^{(l)}\_{i,j} + a^{(l)}\_{j} \ \delta^{(l+1)}\_{i}$ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)} \ (a^{(l)})^T$

7. $D^{(l)}\_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}\_{i,j} + \lambda\Theta^{(l)}\_{i,j}\right) \text{ if } j \neq 0$

8.  $D^{(l)}\_{i,j} := \dfrac{1}{m}\Delta^{(l)}\_{i,j} \text{ if } j = 0$

---
### Unrolling Parameters
In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

```
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

---
### Gradient Checking
Gradient checking will assure that our backpropagation works as intended.

$$
\begin{align}
  \dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
\end{align}
$$

With multiple theta matrices, we can approximate the derivative **with respect to $\Theta\_j$**as follows:

$$
\begin{align}
  \dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}
\end{align}
$$

The professor Andrew usually uses the value $\epsilon = 10^{-4}$

```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

**Note**: Once you've verified once that your backpropagation algorithm is correct, then you don't need to compute gradApprox again. The code to compute gradApprox is very slow.

---
### Randomization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly.

Instead we can randomly initialize our weights between $[-\epsilon, \epsilon]$:
```
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.
% rand(x,y) will initialize a matrix of random real numbers between 0 and 1

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

---
### Putting it together
First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers total.

* Number of input units = dimension of features $x^{(i)}$
* Number of output units = number of classes
* Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
* Defaults: 1 hidden layer. If more than 1 hidden layer, then the same number of units in every hidden layer.

**Training a Neural Network**

1. Randomly initialize the weights
2. Implement forward propagation to get $h\_\theta(x^{(i)})$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.
