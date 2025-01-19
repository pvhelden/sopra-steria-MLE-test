# ML Engineer Take Home Test - Sopra Steria

Philémon van Helden - 19/01/2025

## Requirements

- Python 3.11+
- Libraries found in requirements.txt

## Installation

1. Open a terminal (or command prompt) in the project directory.
2. Run `python -m venv {path/to/venv}` to create a new virtual environment.
3. Activate it.
4. Run `pip install -r requirements.txt` to install all required libraries.

## Creating a model

1. Open a terminal (or command prompt) in the project directory.
2. Run `python ./model_train.py`.

## Running

1. Open a terminal (or command prompt) in the project directory.
2. Run `python -m uvicorn app:app --reload` to run a local server.
3. Open a browser and navigate to `http://127.0.0.1:8000`.
4. Fill the form with data from the house whose price you want to predict.
5. Press the `Predict` button to get a price prediction.

---

## Technologies and Tools

### Python

Chosen as the primary programming language due to its ease of use, prolific libraries for machine learning, and
strong community support.

### FastAPI

Chosen for its simplicity, automatic documentation, and asynchronous capabilities. Also provides high performance whilst
remaining lightweight.

### Jinja2

Adopted for server-side HTML templating in combination with FastAPI. Chosen for its easy rendering of dynamic data on
the frontend.

### Scikit-learn

Chosen for its extensive and complete capabilities regarding machine learning. Provides not only data processing through
pipelines and transformers, but also a wide variety of machine learning models.

For this study, we evaluated the following algorithms:

- LinearRegression
- GammaRegressor
- PoissonRegressor
- TweedieRegressor
- XGBRegressor (Based on RandomForestRegressor)
- SVR
- MLPRegressor

These were chosen for relying on fundamentally different principles and assumptions.

### Mean Absolute Error

Chosen as an evaluation metric for its straightforward interpretation, allowing the client to directly understand the
performance of the model.

## Feature engineering

### One Hot Encoding

The `ocean_proximity` column contains categorical data, which can affect the performance of models. By using a One Hot
Encoder, we can convert it into a column for each category, whose values then become Boolean.

### Scaling

In the given data set, we observe that the variables span over very different scales. This can affect the models'
performance by overweighting some variables. To circumvent this effect, we standardise each column to an average of zero
and a standard deviation of one.

### New columns

We derived two new columns from the given data set:

1. `people_per_room` obtained by dividing the `population` column by the `total_rooms` column.
2. `people_per_household` obtained by dividing the `population` column by the `households` column.

---

## Results

This table summarises results obtained with the preprocessed data, but without the added columns:

| Algorithm        |          Mean MAE | Median MAE |        STD of MAE |
|------------------|------------------:|-----------:|------------------:|
| LinearRegression | 1,243,008,242,754 |     52,711 | 3,729,024,569,310 |
| GammaRegressor   |            67,870 |     67,392 |            14,205 |
| PoissonRegressor |            53,976 |     54,849 |            11,843 |
| TweedieRegressor |            68,970 |     71,730 |            11,719 |
| MLPRegressor     |           123,209 |    121,317 |            42,457 |
| SVR              |            89,737 |     88,392 |            20,176 |
| XGBRegressor     |            46,848 |     45,871 |            13,210 |

### Feature engineering

This table summarises the performances with the new columns:

| Algorithm        |          Mean MAE | Median MAE |        STD of MAE |
|------------------|------------------:|-----------:|------------------:|
| LinearRegression | 1,253,801,657,445 |     52,738 | 3,761,404,813,288 |
| GammaRegressor   |            67,849 |     67,382 |            14,201 |
| PoissonRegressor |            54,011 |     54,880 |            11,854 |
| TweedieRegressor |            68,995 |     71,713 |            11,679 |
| MLPRegressor     |           123,198 |    122,931 |            42,713 |
| SVR              |            89,734 |     88,389 |            20,170 |
| XGBRegressor     |            44,108 |     42,593 |            11,510 |

We can see that the only algorithm for which the performances increase sensibly is the XGBRegressor.

#### LinearRegression

This algorithm assumes independence between variables and linearity between input and target variables.
Based on the data, we did not expect it to perform. It was used mostly as a baseline instead.

As expected, it gives extremely poor results in terms of mean MAE. However, the median is comparable to other
algorithms, which suggests that the mean and standard deviation are heavily influenced by outliers.

#### GLMs (GammaRegressor, PoissonRegressor, TweedieRegressor)

Generalised linear models don't assume linearity between input and target variables. However, they still assume
independence. We tested the three options given by scikit-learn (Gamma, Poisson, Tweedie).

Accordingly, the mean MAE and standard deviation are drastically reduced as compared to the LinearRegression.
Among these three, the best results are achieved with the Poisson, with the other two being approximately equal.

#### MLPRegressor

The Multi-Layer Perceptron Regressor is a neural-network-based algorithm, which means there is no assumption on the
distribution of the variables or their independence.

Its performances are largely worse than others, especially when looking at the median MAE.

#### Support Vector Regression (SVR)

SVMs don't make assumptions on the distribution of the variables either.

#### XGBRegressor

Expected to be the most appropriate machine learning algorithm for this task, based on its ability to handle complex,
non-linear relationships and interactions between features. This is further proved based on results obtained in the
`choose_best_model` function.

Not only is its predictive performance better on average, it is also one of the most efficient algorithms tested,
allowing for more testing and optimisation in a given time frame.

### Hyperparameter tuning

This part focuses on XGBRegressor only.

We used a Coarse-to-Fine Strategy for tuning hyperparameter, by starting with a wide range for each parameter, and
sequentially narrowing them based on the previous results.

This brings the current best performances of the model to:

- Mean MAE: 28,569
- Median MAE: 28,715

## Future Improvements

### Technologies and Tools

The project could be dockerized, depending on its possible applications.
Since the only dependency is Python, and its libraries are listed in the `requirements.txt`, it should be a
straightforward process.

### Hyperparameter tuning

For SVR, we only tested the default kernel. Since kernels have a strong impact on performance, we should explore the
other kernels. They are however the most time-consuming we tested, which makes it hard to use hyperparameter tuning.

For MLPRegressor, it should be noted that hyperparameter tuning is particularly important in neural networks, but
is very time-consuming. This could be further explored on the cloud or given more time.


### Plots

It would be interesting to generate plots and figures at certain steps of the process. An example would be a scatter
plot of the actual prices versus predicted price for each algorithm. This would provide a more refined view on the
performances and biases of the respective algorithms (such as highlighting outliers).
