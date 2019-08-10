+++
author = "Guocheng Wei"
categories = ["Coursera Notes"]
tags = ["Machine Learning"]
date = "2019-06-28"
description = "Multivariate Linear Regression, Normal Equation, Octave Tutorial"
featured = "ml_stanford_2.png"
featuredalt = "ml stanford thumbnail"
featuredpath = "date"
linktitle = ""
title = "Week 2 - Machine Learning"
type = "post"

+++

---
### Mutiple Features
Linear regression with multiple variables is also known as **multivariate linear regression**.

The notation for equations:

$$ x_j^{(i)} = \text{value of feature } j \text{ in the }i^{th}\text{ training example} $$

$$ x^{(i)} = \text{the input (features) of the }i^{th}\text{ training example} $$

$$ m = \text{the number of training examples} $$

$$ n = \text{the number of features} $$

The multivariable form of the hypothesis function:

$$ h_\theta (x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n $$

Assume $$ x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$$

Then, multivariable hypothesis function can be concisely represented as:

$$ h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x $$

---
### Gradient Descent for Multiple Variables
Repeat until convergence: {
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)}) - y^{(i)}) \cdot x\_j^{(i)} \qquad \text{for j := 0...n}$$
}

---
### Feature Scaling and Mean Normalization
We can speed up gradient descent by having each of our input values in roughly the same range. This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

**Feature scaling** involves dividing the input values by the range (i.e. the maximum value minus the minimum value or the standard deviation) of the input variable, resulting in a new range of just 1.

**Mean normalization** involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

So:

$$x\_{i} := \frac{x\_{i} - \mu\_{i}}{s\_{i}}$$

$\mu\_{i}$ is the average of all values for feature $(i)$
$s\_{i}$ is  the range of values (max - min), is the standard deviation.

---
### Debugging Gradient Descent by Learning Rate
Make a plot with number of iterations on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

{{< fancybox path="/img/2019/06" file="check_learning_rate.png" caption="Plot With Number of Iterations" gallery="Note Images" >}}

**If $\alpha$ is too small: slow convergence.** \\
**If $\alpha$ is too large: ￼may not decrease on every iteration and thus may not converge.**

---
### Polynominal Regression
We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

---
### Normal Equation

$$\theta = (X^TX)^{-1}X^Ty$$

A method of finding the optimum theta **without iteration**. \\
**No need** to do feature scaling with the normal equation.

[Proof of the normal equation](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression) 

| Gradient Descent           | Normal Equation                          |
|:--------------------------:|:----------------------------------------:|
| Need to choose alpha       | No need to choose alpha                  |
| Needs many iterations      | No need to iterate                       |
| $O(kn^2)$                  | $O(n^3)$ need to calculate $(X^TX)^{-1}$ |
| Works well when n is large | Slow if n is very large                  |

#### Noninvertability
If $X^TX$ is **noninvertible**, the common causes might be having :

* **Redundant features**, where two features are very closely related (i.e. they are linearly dependent) \\
* **Too many features** (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

When implementing the normal equation in octave we want to use the **'pinv'** function rather than 'inv.'
