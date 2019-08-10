+++
author = "Guocheng Wei"
categories = ["Coursera Notes"]
tags = ["Machine Learning"]
date = "2019-06-27"
description = "Notes of Week 1 Machine Learning Coursera Course by Stanford University"
featured = "ml_stanford_1.png"
featuredalt = "ml stanford thumbnail"
featuredpath = "date"
linktitle = ""
title = "Week 1 - Machine Learning"
type = "post"

+++

---
### The Hypothesis Function
$$\hat{y} = h_\theta(x) = \theta_0 + \theta_1 x$$

---
### Cost Function
To measure the accuracy of the hypothesis function. This takes an average (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's compared to the actual output y's.

$$J(\theta_0, \theta_1) = \dfrac{1}{2m} \displaystyle \sum _{i=1}^m  \left( \hat{y}_i- y_i \right)^2 = \dfrac{1}{2m} \displaystyle \sum _{i=1}^m \left (h _\theta(x_i) - y_i \right)^2$$

Break it apart:

> $h_\theta (x_i) - y_i$ is the difference between the predicted value and the actual value.

> $\bar{x}$ is the mean of all $\left (h_\theta (x_i) - y_i \right)^2$

>$J(\theta_0, \theta_1) = \frac{1}{2}\bar{x} $

The function is also called **Square Error Function**, or **Mean squared error**

The mean is halfed as a convenience for the computation of **the gradient descent**, as the derivative of the square function will cancel out the $\frac{1}{2}$ term

#### Intuition Ⅰ
To simplify the visualization of the cost function, assume $\theta_0$ is 0, which means the cost function is $J(\theta_1)$

{{< fancybox path="/img/2019/06" file="intuition_1.png" caption="Intuition of theta to find the minimal cost function" gallery="Note Images" >}}

#### Intuition Ⅱ
If the $\theta_0$ is not 0, then the **contour plot** will look like this:

{{< fancybox path="/img/2019/06" file="contour_plot_3d.png" caption="3D Contour Plot" gallery="Note Images" >}}

If projected onto a 2d plot

{{< fancybox path="/img/2019/06" file="contour_plot_2d.png" caption="2D Contour Plot" gallery="Note Images" >}}

Key features:

* Every point on the same 'circle' has the same $J(\theta_0, \theta_1)$
* The center of the inner most 'circle' is the minimal cost function value.

---
### Gradient Descent
To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

{{< fancybox path="/img/2019/06" file="gradient_descent_example.png" caption="Example of Gradient Descent" gallery="Note Images" >}}

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the **learning rate**.

As shown above, two points are next to each other, but their local minimal cost function value are different.

The gradient descent algorithm is:

> repear until convergence:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$
where $j = 0,1$ represents the feature index number

At each iteration j, one should simultaneously update the parameters $\theta_0, \theta_1, \theta_2, ...$. (**Simultaneous Update**)

#### Intuition Ⅰ
Simplied gradient descent with $\theta_0$ is 0:

> repear until convergence:
$$\theta_1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_1)$$

**When the slope is negative, the value of $\theta_1$ ​increases.**

**When it is positive, the value of $\theta_1$ decreases.**

{{< fancybox path="/img/2019/06" file="gradient_descent_intuition_slope.png" caption="How slope affects the movement of theta" gallery="Note Images" >}}

#### Intuition Ⅱ
We should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time. 

Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

{{< fancybox path="/img/2019/06" file="gradient_descent_intuition_step.png" caption="Step size affects the convergence" gallery="Note Images" >}}

#### Intuition Ⅲ
**Gradient descent can converge to a local minimum, even with the learning rate is fixed.**

When approaching the local minimum, slope value is smaller, and so the step size is smaller.

---
### Gradient Descent For Linear Regression

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mtext>repeat until convergence:&#xA0;</mtext>
        <mo fence="false" stretchy="false">{</mo>
      </mtd>
      <mtd />
    </mtr>
    <mtr>
      <mtd>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>0</mn>
        </msub>
        <mo>:=</mo>
      </mtd>
      <mtd>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>0</mn>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03B1;<!-- α --></mi>
        <mfrac>
          <mn>1</mn>
          <mi>m</mi>
        </mfrac>
        <munderover>
          <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
          </mrow>
        </msub>
        <mo stretchy="false">)</mo>
        <mo>&#x2212;<!-- − --></mo>
        <msub>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
          </mrow>
        </msub>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>1</mn>
        </msub>
        <mo>:=</mo>
      </mtd>
      <mtd>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>1</mn>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03B1;<!-- α --></mi>
        <mfrac>
          <mn>1</mn>
          <mi>m</mi>
        </mfrac>
        <munderover>
          <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <mfenced open="(" close=")">
          <mrow>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>h</mi>
              <mi>&#x03B8;<!-- θ --></mi>
            </msub>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <mo>&#x2212;<!-- − --></mo>
            <msub>
              <mi>y</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
          </mrow>
        </mfenced>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
      <mtd />
    </mtr>
  </mtable>
</math>

**Batch gradient descent**: looks at every example in the entire training set on every step
