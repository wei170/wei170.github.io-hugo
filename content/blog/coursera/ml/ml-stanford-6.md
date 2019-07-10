+++
author = "Guocheng Wei"
categories = ["Coursera Notes"]
tags = ["Machine Learning"]
date = "2019-07-06"
description = "Model Selection, ML Diagnostic Method, DNN Diagnostic Method, Error Metrics for Skewed Classes, Vanishing / Exploding Gradient"
featured = "ml_stanford_6.png"
featuredalt = "ml stanford thumbnail"
featuredpath = "date"
linktitle = ""
title = "Week 6 - Machine Learning"
type = "post"

+++

---
### Deciding What to Try Next

Errors in your predictions can be troubleshooted by:

* Getting more training examples
* Trying smaller sets of features
* Trying additional features
* Trying polynomial features
* Increasing or decreasing $\lambda$

---
### Model Selection and Train/Validation/Test Sets

#### Test Error

$$
J\_{test}(\Theta) = \dfrac{1}{2m\_{test}} \sum\_{i=1}^{m\_{test}}(h\_\Theta(x^{(i)}\_{test}) - y^{(i)}\_{test})^2
$$

* Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis.
* The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than any other data set.

#### Use of the CV set
To solve this, we can introduce a third set, the **Cross Validation Set**, to serve as an intermediate set that we can train d with. Then our test set will give us an accurate, **non-optimistic error**.

Model selection:

1. Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $J\_{test}(\Theta^{(d)})$ (d = theta from polynomial with lower error);

![CV Model](/img/2019/07/cv_model.png)

---
### Diagnosing Bias vs. Variance

The training error will tend to **decrease** as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to **decrease** as we increase d up to a point, and then it will **increase** as d is increased, forming a convex curve.

<img src="/img/2019/07/bias_vs_variance.png" alt="Bias vs. Variance" style="width:300px;"/>

---
### Regularization and Bias/Variance

* Large $\lambda$: High bias (underfitting)
* Intermediate $\lambda$: just right
* Small $\lambda$: High variance (overfitting)

<img src="/img/2019/07/large_small_lambda.png" alt="Large/Small Lambda" style="width:300px;"/>

---
**In order to choose the model and the regularization $\lambda$**, we need:

1. Create a list of lambdas (i.e. $\lambda \in\\{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24\\}$);

2. Create a set of models with different degrees or any other variants.

3. Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$.

4. Compute the cross validation error using the learned $\lambda$ (computed with $\lambda$) on the $J\_{CV}(\Theta)$ without regularization or $\lambda = 0$.

5. Select the best combo that produces the lowest error on the cross validation set.

6. Using the best combo $\Theta$ and $\lambda$, apply it on $J\_{test}(\Theta)$ to see if it has a good generalization of the problem.

---
### Learning Curves

**With high bias**:

* **Low training set size**: causes $J\_{train}(\Theta)$ to be low and $J\_{CV}(\Theta)$ to be high.

* **Large training set size**: causes both $J\_{train}(\Theta)$ and $J\_{CV}(\Theta)$ to be high with $J\_{train}(\Theta) \approx J\_{CV}(\Theta)$.

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

<img src="/img/2019/07/learning_curve_high_bias.png" alt="Learning Curve with High Bias" style="width:300px;"/>

**With high variance**:

* Low training set size: $J\_{train}(\Theta)$ will be low and $J\_{CV}(\Theta)$ will be high.

* Large training set size: $J\_{train}(\Theta)$ increases with training set size and $J\_{CV}(\Theta)$ continues to decrease without leveling off. Also, $J\_{train}(\Theta) < J\_{CV}(\Theta)$ but the difference between them remains significant.

<img src="/img/2019/07/learning_curve_high_variance.png" alt="Learning Curve with High Variance" style="width:300px;"/>

---
### Deciding What to Do Next Revisited

| Method                           |    Usage            |
| -------------------------------- | ------------------- |
| Getting more training examples   | Fixes high variance |
| Trying smaller sets of features  | Fixes high variance |
| Adding features                  | Fixes high bias     |
| Adding polynomial features       | Fixes high bias     |
| Decreasing $\lambda$             | Fixes high bias     |
| Increasing $\lambda$             | Fixes high variance |

---
### Diagnosing Deep Neural Networks
* A neural network with **fewer parameters is prone to underfitting**. It is also **computationally cheaper**.
* A large neural network with **more parameters is prone to overfitting**. It is also **computationally expensive**. In this case you can use regularization (increase $\lambda$) to address the overfitting.


---
### Error Metrics for Skewed Classes

It is sometimes difficult to tell whether a reduction in error is actually an improvement of the algorithm.

* For example: In predicting a cancer diagnoses where 0.5% of the examples have cancer, we find our learning algorithm has a 1% error. However, if we were to simply classify every single example as a 0, then our error would reduce to 0.5% even though we did not improve the algorithm.

This usually happens with **skewed classes**.

For this we can use **Precision/Recall**

![Error Table](/img/2019/07/error_table.jpg)

**Precision**:
$$
\dfrac{\text{True Positives}}{\text{Total number of predicted positives}}
= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False Positives}}
$$

**Recall**:
$$
\dfrac{\text{True Positives}}{\text{Total number of actual positives}}
= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False Negatives}}
$$

By setting the **threshold higher** (i.e, $h\_\theta(x) \geq 0.7$ predict 1), you can get a **confident** prediction, **higher precision** but **lower recall**

By setting the **threshold lower** (i.e, $h\_\theta(x) \geq 0.3$ predict 1), you can get a **safe** prediction,  **higher recall** but **lower precision**

Use **F score** to leverage the two metrics:
$$\text{F score} = 2 \frac{PR}{P+R}$$
