---
author:
- Kaleb Cervantes
authors:
- Kaleb Cervantes
execute:
  warning: false
title: Regression in Python
toc-title: Table of contents
---

# Intro

Last semester, we went over how to do regression in R. This is fairly
straightfoward as we just use the functions `lm` and `glm` to make the
models. In R, we simply needed the data, and the formula. However this
is more complicated in Python.

First I will want to import the following libraries:

-   `sklearn` for the linear and logistic regression functions

-   `numpy` for matrix stuff

-   `pandas` for viewing dataframes.

::: {.cell execution_count="1"}
``` {.python .cell-code}
import sklearn.linear_model as skl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
```
:::

I will also be using the auction verification dataset from UCI
repository. This is because it has both a numeric and binary response
for linear and logistic regression respectively. The head of the data is
given in the following three tables. The first two are predictors, and
the third are responses.

::: {.cell execution_count="2"}
``` {.python .cell-code}
auction_data = pd.read_csv("data.csv")

auction_data.iloc[0:4, 0:4]
```

::: {.cell-output .cell-output-display execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>process.b1.capacity</th>
      <th>process.b2.capacity</th>
      <th>process.b3.capacity</th>
      <th>process.b4.capacity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell execution_count="3"}
``` {.python .cell-code}
auction_data.iloc[0:4, 4:7]
```

::: {.cell-output .cell-output-display execution_count="3"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>property.price</th>
      <th>property.product</th>
      <th>property.winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell execution_count="4"}
``` {.python .cell-code}
auction_data.iloc[0:4, 7:9]
```

::: {.cell-output .cell-output-display execution_count="4"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>verification.result</th>
      <th>verification.time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>163.316667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>200.860000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>154.888889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>108.640000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

# Preparing The Data

It may help to understand the dimensions of the data.

::: {.cell execution_count="5"}
``` {.python .cell-code}
auction_data.shape
```

::: {.cell-output .cell-output-display execution_count="5"}
    (2043, 9)
:::
:::

Since this data has a fairly large amount of observations, it may help
to split the data into training and test sets. For this, it may help to
identify the following responses:

-   `verification.result` for logistic regression

-   `verification.time` for linear regression

## Handling Predictors

It may help to do some exploration with the predictors first. Now it may
be important to note that --- although all of the predictors are stored
as integers --- there may be some categorical data. From reading the
documentation, this seems to be the case for the following:

-   `property.product` --- product code for product currently being
    verified.

-   `property.winner` --- `0` if price was verified, otherwise bidder
    code for bidder currently being verified

The above columns will be transformed using dummy variables, with their
default value `0` being ignored. Conveniently, the responses are also
the last two columns in the dataframe. This means that our predictors
are the first seven columns. Since indeces in Python begin at 0, the
following code chunk will extract the predictors and add dummy
variables.

::: {.cell execution_count="6"}
``` {.python .cell-code}
X = pd.get_dummies(
    auction_data.iloc[:, 0:7],
    columns= ["property.product", "property.winner"],
    drop_first=True
)
```
:::

It may also help to note that the dimensions have changed.

::: {.cell execution_count="7"}
``` {.python .cell-code}
X.shape
```

::: {.cell-output .cell-output-display execution_count="7"}
    (2043, 14)
:::
:::

Even though most of these are dummy variables, they will add more
coefficients to the regression model.

## Splitting the Data

Now that the predictors have been handled, we can split the data. This
process may need to be repeated later on depending on the circumstances.
This is actually one of the areas where Python is more straightfoward
than --- base --- R.

the function `sklearn.model_selection.train_test_split` splits the into
training and test data. The first inputs for this function are the
predictor matrix and response vector --- or vectors if splitting for
multiple responses. By default this function does a 75-25 split for
training and testing. We can specify the split by putting the desired
ratio for either group in the parameters `test_size` or `train_size`.
There is also the function `random_state` which can be used to specify
the seed set for random sample used. Dr. Kerr used the R equivelent for
reproducabiliity so I intend on doing the same.

::: {.cell execution_count="8"}
``` {.python .cell-code}
y_lin = auction_data["verification.time"]
y_log = auction_data["verification.result"]

(
    X_train, X_test,
    y_lin_train, y_lin_test,
    y_log_train, y_log_test
) = train_test_split(
    X, y_lin, y_log,
    test_size = 0.4,
    random_state = 2022
)
```
:::

Now that the data has been split, we can finally fit our model.

# Linear Models

Similar to `lm` in R, `sklearn.linear_model.LinearRegression` is an
object for our linear model. The function `fit` will be used to actually
fit the model.

In order to see the $R^2$ coefficient, we use the function `score`. We
will first check this for the training data.

::: {.cell execution_count="9"}
``` {.python .cell-code}
lm1 = (
    skl
    .LinearRegression()
    .fit(X_train, y_lin_train)
)

lm1.score(X_train, y_lin_train)
```

::: {.cell-output .cell-output-display execution_count="9"}
    0.6605544724598729
:::
:::

From this it appears that `lm1` only accounts for about 66% of the
variability in the model. Now we try to see what this value would be for
the test data.

::: {.cell execution_count="10"}
``` {.python .cell-code}
lm1.score(X_test, y_lin_test)
```

::: {.cell-output .cell-output-display execution_count="10"}
    0.6182826928671714
:::
:::

From this it appears that only about 62% of the variability in the test
data is accounted for by the model. This is not perfect, but given the
$R^2$ for the training data this seems ok.

If we want to see the coefficients of the model, we have to access the
attributes `coef_` for the predictors and `intercept_` for the
intercept.

::: {.cell execution_count="11"}
``` {.python .cell-code}
lm1.intercept_
```

::: {.cell-output .cell-output-display execution_count="11"}
    8190.3895882599245
:::
:::

::: {.cell execution_count="12"}
``` {.python .cell-code}
lm1.coef_
```

::: {.cell-output .cell-output-display execution_count="12"}
    array([  5850.77182834,     84.72484738,  -1143.55239713,   2362.55568998,
              -48.5921103 ,   9920.36748465, -10062.44991919,  -7612.87468259,
            -5228.64958719,  -9800.90224685,  -5832.05652052,   1195.38994045,
             1986.24482746,    691.72620791])
:::
:::

In R, the functions `lm`, `summary.lm`, and `plot.lm` do most of the
above and much more. Unfortunately, Python doesn't allow for diagnostics
to be done as easily. There are other packages beside sklearn that do
them, but they are not as efficient as doing them in R. This is why ---
although I prefer Python as a language over R --- R is much better for
regression and diagnostics.

# Logistic Regression

When we split the data, we included both responses. Luckily this means
that the splitting portion has been done for the logistic regression.
Doing this is very similar to linear regression. It is important to note
that by default, logistic regression in python applies an l2 penalty.
This is similar to ridge regression. To remove the penalty, I set the
parameter `penalty` to `"none"`.

::: {.cell execution_count="13"}
``` {.python .cell-code}
glm1 = (
    skl
    .LogisticRegression("none")
    .fit(X_train, y_log_train)
)
```
:::

Now that the logistical model is fit, I will use `score` to check
accuracy. Here `score` returns the number of correct predictions divided
by the total number of predictions.

::: {.cell execution_count="14"}
``` {.python .cell-code}
glm1.score(X_train, y_log_train)
```

::: {.cell-output .cell-output-display execution_count="14"}
    0.9069387755102041
:::
:::

::: {.cell execution_count="15"}
``` {.python .cell-code}
glm1.score(X_test, y_log_test)
```

::: {.cell-output .cell-output-display execution_count="15"}
    0.902200488997555
:::
:::

The logistic model predicted with an accuracy of about 90% for both the
test and training sets.

Unfortunately, a lot of the model diagnostics still have to be manually
done as there are not simple `plot.glm` or `summary.glm` in
`scikitlearn`.

# `statsmodels`

`statsmodels` is a python library that allows users to use R-style
formulas in Python. This would be useful for mixed model stuff, but in
this document I will mostly use it to read model summaries.

Since we are using this library, we will use the function `add_constant`
to add the constant term to the training data matrix.

::: {.cell execution_count="16"}
``` {.python .cell-code}
import statsmodels.api as sm

from statsmodels.tools.tools import add_constant
```
:::

## Linear Regression Tables

Now we will view the model summary tables. There is a third table, but
we did not go over what it shows in STAT 632. As such I will only show
the first two.

::: {.cell execution_count="17"}
``` {.python .cell-code}
X_train_sm = add_constant(X_train)

lm1_summary_tables = (
    sm
    .OLS(y_lin_train, X_train_sm)
    .fit()
    .summary()
    .tables
)

lm1_summary_tables[0]
```

::: {.cell-output .cell-output-display execution_count="17"}
```{=html}
<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>verification.time</td> <th>  R-squared:         </th> <td>   0.661</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.657</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   168.2</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 May 2022</td>  <th>  Prob (F-statistic):</th> <td>7.69e-272</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:40:08</td>      <th>  Log-Likelihood:    </th> <td> -12425.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1225</td>       <th>  AIC:               </th> <td>2.488e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1210</td>       <th>  BIC:               </th> <td>2.496e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    14</td>       <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>     <th>                     </th>     <td> </td>    
</tr>
</table>
```
:::
:::

::: {.cell execution_count="18"}
``` {.python .cell-code}
lm1_summary_tables[1]
```

::: {.cell-output .cell-output-display execution_count="18"}
```{=html}
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td> 8190.3896</td> <td> 2227.571</td> <td>    3.677</td> <td> 0.000</td> <td> 3820.059</td> <td> 1.26e+04</td>
</tr>
<tr>
  <th>process.b1.capacity</th> <td> 5850.7718</td> <td>  261.338</td> <td>   22.388</td> <td> 0.000</td> <td> 5338.046</td> <td> 6363.498</td>
</tr>
<tr>
  <th>process.b2.capacity</th> <td>   84.7248</td> <td>  226.182</td> <td>    0.375</td> <td> 0.708</td> <td> -359.028</td> <td>  528.478</td>
</tr>
<tr>
  <th>process.b3.capacity</th> <td>-1143.5524</td> <td>  643.268</td> <td>   -1.778</td> <td> 0.076</td> <td>-2405.598</td> <td>  118.493</td>
</tr>
<tr>
  <th>process.b4.capacity</th> <td> 2362.5557</td> <td>  385.525</td> <td>    6.128</td> <td> 0.000</td> <td> 1606.185</td> <td> 3118.926</td>
</tr>
<tr>
  <th>property.price</th>      <td>  -48.5921</td> <td>   27.998</td> <td>   -1.736</td> <td> 0.083</td> <td> -103.523</td> <td>    6.339</td>
</tr>
<tr>
  <th>property.product_2</th>  <td> 9920.3675</td> <td>  547.889</td> <td>   18.107</td> <td> 0.000</td> <td> 8845.449</td> <td>  1.1e+04</td>
</tr>
<tr>
  <th>property.product_3</th>  <td>-1.006e+04</td> <td>  636.074</td> <td>  -15.820</td> <td> 0.000</td> <td>-1.13e+04</td> <td>-8814.520</td>
</tr>
<tr>
  <th>property.product_4</th>  <td>-7612.8747</td> <td>  655.433</td> <td>  -11.615</td> <td> 0.000</td> <td>-8898.785</td> <td>-6326.964</td>
</tr>
<tr>
  <th>property.product_5</th>  <td>-5228.6496</td> <td>  687.055</td> <td>   -7.610</td> <td> 0.000</td> <td>-6576.601</td> <td>-3880.698</td>
</tr>
<tr>
  <th>property.product_6</th>  <td>-9800.9022</td> <td>  598.886</td> <td>  -16.365</td> <td> 0.000</td> <td> -1.1e+04</td> <td>-8625.932</td>
</tr>
<tr>
  <th>property.winner_1</th>   <td>-5832.0565</td> <td> 1137.179</td> <td>   -5.129</td> <td> 0.000</td> <td>-8063.118</td> <td>-3600.995</td>
</tr>
<tr>
  <th>property.winner_2</th>   <td> 1195.3899</td> <td>  780.814</td> <td>    1.531</td> <td> 0.126</td> <td> -336.509</td> <td> 2727.289</td>
</tr>
<tr>
  <th>property.winner_3</th>   <td> 1986.2448</td> <td>  765.638</td> <td>    2.594</td> <td> 0.010</td> <td>  484.120</td> <td> 3488.370</td>
</tr>
<tr>
  <th>property.winner_4</th>   <td>  691.7262</td> <td>  966.348</td> <td>    0.716</td> <td> 0.474</td> <td>-1204.177</td> <td> 2587.629</td>
</tr>
</table>
```
:::
:::

From this we can see that both of the categorical variables have
significant and insignificant levels. I will check Dr. Kerr's notes on
how to handle these.

The variable `process.b2.capacity` is not significant at any resonable
level of $\alpha$.

We also notice that `property.price` and `process.b3.capacity` are not
significant at the $\alpha = 0.05$ level, but would be significant at
the $\alpha = 0.1$ level.

## Logistic Regression Table

::: {.cell execution_count="19"}
``` {.python .cell-code}
(
    sm
    .Logit(y_log_train, X_train_sm)
    .fit()
    .summary()
    .tables[1]
)
```

::: {.cell-output .cell-output-stdout}
    Optimization terminated successfully.
             Current function value: 0.249993
             Iterations 8
:::

::: {.cell-output .cell-output-display execution_count="19"}
```{=html}
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td>  -11.7327</td> <td>    1.591</td> <td>   -7.375</td> <td> 0.000</td> <td>  -14.851</td> <td>   -8.615</td>
</tr>
<tr>
  <th>process.b1.capacity</th> <td>   -1.3342</td> <td>    0.208</td> <td>   -6.405</td> <td> 0.000</td> <td>   -1.742</td> <td>   -0.926</td>
</tr>
<tr>
  <th>process.b2.capacity</th> <td>   -0.3079</td> <td>    0.135</td> <td>   -2.287</td> <td> 0.022</td> <td>   -0.572</td> <td>   -0.044</td>
</tr>
<tr>
  <th>process.b3.capacity</th> <td>   -0.0702</td> <td>    0.328</td> <td>   -0.214</td> <td> 0.831</td> <td>   -0.713</td> <td>    0.573</td>
</tr>
<tr>
  <th>process.b4.capacity</th> <td>   -0.0463</td> <td>    0.238</td> <td>   -0.195</td> <td> 0.846</td> <td>   -0.512</td> <td>    0.419</td>
</tr>
<tr>
  <th>property.price</th>      <td>    0.1464</td> <td>    0.021</td> <td>    7.026</td> <td> 0.000</td> <td>    0.106</td> <td>    0.187</td>
</tr>
<tr>
  <th>property.product_2</th>  <td>   -0.2652</td> <td>    0.396</td> <td>   -0.669</td> <td> 0.503</td> <td>   -1.042</td> <td>    0.511</td>
</tr>
<tr>
  <th>property.product_3</th>  <td>    1.1049</td> <td>    0.373</td> <td>    2.960</td> <td> 0.003</td> <td>    0.373</td> <td>    1.837</td>
</tr>
<tr>
  <th>property.product_4</th>  <td>    1.3371</td> <td>    0.410</td> <td>    3.257</td> <td> 0.001</td> <td>    0.533</td> <td>    2.142</td>
</tr>
<tr>
  <th>property.product_5</th>  <td>   -0.8076</td> <td>    0.434</td> <td>   -1.863</td> <td> 0.063</td> <td>   -1.658</td> <td>    0.042</td>
</tr>
<tr>
  <th>property.product_6</th>  <td>    1.3737</td> <td>    0.394</td> <td>    3.488</td> <td> 0.000</td> <td>    0.602</td> <td>    2.146</td>
</tr>
<tr>
  <th>property.winner_1</th>   <td>    5.1913</td> <td>    0.794</td> <td>    6.542</td> <td> 0.000</td> <td>    3.636</td> <td>    6.747</td>
</tr>
<tr>
  <th>property.winner_2</th>   <td>    2.1573</td> <td>    0.294</td> <td>    7.329</td> <td> 0.000</td> <td>    1.580</td> <td>    2.734</td>
</tr>
<tr>
  <th>property.winner_3</th>   <td>   -0.9584</td> <td>    0.498</td> <td>   -1.926</td> <td> 0.054</td> <td>   -1.934</td> <td>    0.017</td>
</tr>
<tr>
  <th>property.winner_4</th>   <td>    0.1983</td> <td>    0.430</td> <td>    0.461</td> <td> 0.644</td> <td>   -0.644</td> <td>    1.041</td>
</tr>
</table>
```
:::
:::

# Reduced Models

This is a bit more complicated in Python than it is in R. In R, we were
able to remove predictors in the formula. In Python, we have to remove
the corresponding columns in the dataframe or matrix. The following will
remove the columns and refit the model.

## Reduced Linear Model

::: {.cell execution_count="20"}
``` {.python .cell-code}
X_train_lin_reduced = X_train.drop(
    ["process.b2.capacity", "process.b3.capacity", "property.price"],
    1
)
X_test_lin_reduced = X_test.drop(
    ["process.b2.capacity", "process.b3.capacity", "property.price"],
    1
)

lm2 = (
    skl
    .LinearRegression()
    .fit(X_train_lin_reduced, y_lin_train)
)

lm2.score(X_train_lin_reduced, y_lin_train)
```

::: {.cell-output .cell-output-display execution_count="20"}
    0.6585338975456794
:::
:::

::: {.cell execution_count="21"}
``` {.python .cell-code}
lm2.score(X_test_lin_reduced, y_lin_test)
```

::: {.cell-output .cell-output-display execution_count="21"}
    0.6124603279912988
:::
:::

From the above, we can see that there does not seem to be a significant
difference in the $R^2$ from reducing the models. We can now look at the
new table for the coefficients.

::: {.cell execution_count="22"}
``` {.python .cell-code}
(
    sm
    .OLS(y_lin_train, add_constant(X_train_lin_reduced))
    .fit()
    .summary()
    .tables[1]
)
```

::: {.cell-output .cell-output-display execution_count="22"}
```{=html}
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td> 3083.6728</td> <td>  491.000</td> <td>    6.280</td> <td> 0.000</td> <td> 2120.370</td> <td> 4046.976</td>
</tr>
<tr>
  <th>process.b1.capacity</th> <td> 5532.9222</td> <td>  228.843</td> <td>   24.178</td> <td> 0.000</td> <td> 5083.949</td> <td> 5981.895</td>
</tr>
<tr>
  <th>process.b4.capacity</th> <td> 2385.9731</td> <td>  385.147</td> <td>    6.195</td> <td> 0.000</td> <td> 1630.345</td> <td> 3141.601</td>
</tr>
<tr>
  <th>property.product_2</th>  <td> 9808.2027</td> <td>  532.179</td> <td>   18.430</td> <td> 0.000</td> <td> 8764.109</td> <td> 1.09e+04</td>
</tr>
<tr>
  <th>property.product_3</th>  <td>-1.013e+04</td> <td>  629.158</td> <td>  -16.101</td> <td> 0.000</td> <td>-1.14e+04</td> <td>-8895.575</td>
</tr>
<tr>
  <th>property.product_4</th>  <td>-7245.2568</td> <td>  630.906</td> <td>  -11.484</td> <td> 0.000</td> <td>-8483.045</td> <td>-6007.469</td>
</tr>
<tr>
  <th>property.product_5</th>  <td>-5565.8221</td> <td>  655.438</td> <td>   -8.492</td> <td> 0.000</td> <td>-6851.741</td> <td>-4279.904</td>
</tr>
<tr>
  <th>property.product_6</th>  <td>-9645.2083</td> <td>  578.390</td> <td>  -16.676</td> <td> 0.000</td> <td>-1.08e+04</td> <td>-8510.452</td>
</tr>
<tr>
  <th>property.winner_1</th>   <td>-6320.4783</td> <td> 1104.305</td> <td>   -5.723</td> <td> 0.000</td> <td>-8487.038</td> <td>-4153.918</td>
</tr>
<tr>
  <th>property.winner_2</th>   <td> 1010.4613</td> <td>  761.877</td> <td>    1.326</td> <td> 0.185</td> <td> -484.281</td> <td> 2505.204</td>
</tr>
<tr>
  <th>property.winner_3</th>   <td> 1719.5593</td> <td>  747.140</td> <td>    2.302</td> <td> 0.022</td> <td>  253.729</td> <td> 3185.390</td>
</tr>
<tr>
  <th>property.winner_4</th>   <td>  487.4516</td> <td>  955.656</td> <td>    0.510</td> <td> 0.610</td> <td>-1387.470</td> <td> 2362.373</td>
</tr>
</table>
```
:::
:::

## Reduced Logistic Model

::: {.cell execution_count="23"}
``` {.python .cell-code}
X_train_log_reduced = X_train.drop(
    ["process.b3.capacity", "process.b4.capacity"],
    1
)
X_test_log_reduced = X_test.drop(
    ["process.b3.capacity", "process.b4.capacity"],
    1
)

glm2 = (
    skl
    .LogisticRegression("none")
    .fit(X_train_log_reduced, y_log_train)
)

glm2.score(X_train_log_reduced, y_log_train)
```

::: {.cell-output .cell-output-display execution_count="23"}
    0.9044897959183673
:::
:::

::: {.cell execution_count="24"}
``` {.python .cell-code}
glm2.score(X_test_log_reduced, y_log_test)
```

::: {.cell-output .cell-output-display execution_count="24"}
    0.9009779951100244
:::
:::

::: {.cell execution_count="25"}
``` {.python .cell-code}
(
    sm
    .Logit(y_log_train, add_constant(X_train_log_reduced))
    .fit()
    .summary()
    .tables[1]
)
```

::: {.cell-output .cell-output-stdout}
    Optimization terminated successfully.
             Current function value: 0.250025
             Iterations 8
:::

::: {.cell-output .cell-output-display execution_count="25"}
```{=html}
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td>  -11.8459</td> <td>    1.535</td> <td>   -7.718</td> <td> 0.000</td> <td>  -14.854</td> <td>   -8.838</td>
</tr>
<tr>
  <th>process.b1.capacity</th> <td>   -1.3453</td> <td>    0.201</td> <td>   -6.709</td> <td> 0.000</td> <td>   -1.738</td> <td>   -0.952</td>
</tr>
<tr>
  <th>process.b2.capacity</th> <td>   -0.3021</td> <td>    0.133</td> <td>   -2.273</td> <td> 0.023</td> <td>   -0.562</td> <td>   -0.042</td>
</tr>
<tr>
  <th>property.price</th>      <td>    0.1458</td> <td>    0.021</td> <td>    7.075</td> <td> 0.000</td> <td>    0.105</td> <td>    0.186</td>
</tr>
<tr>
  <th>property.product_2</th>  <td>   -0.2775</td> <td>    0.389</td> <td>   -0.713</td> <td> 0.476</td> <td>   -1.040</td> <td>    0.485</td>
</tr>
<tr>
  <th>property.product_3</th>  <td>    1.1044</td> <td>    0.373</td> <td>    2.960</td> <td> 0.003</td> <td>    0.373</td> <td>    1.836</td>
</tr>
<tr>
  <th>property.product_4</th>  <td>    1.3415</td> <td>    0.410</td> <td>    3.269</td> <td> 0.001</td> <td>    0.537</td> <td>    2.146</td>
</tr>
<tr>
  <th>property.product_5</th>  <td>   -0.8201</td> <td>    0.422</td> <td>   -1.942</td> <td> 0.052</td> <td>   -1.648</td> <td>    0.008</td>
</tr>
<tr>
  <th>property.product_6</th>  <td>    1.3697</td> <td>    0.393</td> <td>    3.487</td> <td> 0.000</td> <td>    0.600</td> <td>    2.140</td>
</tr>
<tr>
  <th>property.winner_1</th>   <td>    5.1986</td> <td>    0.792</td> <td>    6.563</td> <td> 0.000</td> <td>    3.646</td> <td>    6.751</td>
</tr>
<tr>
  <th>property.winner_2</th>   <td>    2.1678</td> <td>    0.292</td> <td>    7.426</td> <td> 0.000</td> <td>    1.596</td> <td>    2.740</td>
</tr>
<tr>
  <th>property.winner_3</th>   <td>   -0.9585</td> <td>    0.497</td> <td>   -1.929</td> <td> 0.054</td> <td>   -1.933</td> <td>    0.016</td>
</tr>
<tr>
  <th>property.winner_4</th>   <td>    0.1852</td> <td>    0.419</td> <td>    0.442</td> <td> 0.658</td> <td>   -0.635</td> <td>    1.006</td>
</tr>
</table>
```
:::
:::

# Conclusion

Regression can be done in Python and with the use of libraries like
`statsmodels`, may offer the same tools that can be used in R. I think
that with how common Python is, it is worth learning these methods.
However they are not as simple as they are in R.
