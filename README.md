# Weather Data Analysis & Forecasting
The central goal of this learning project is to combine data engineering, database design, classical
statistics, and machine learning (both hand-built and ready-to-use models)
in a single, enterprise-style setup. The project is designed to run entirely on free
tools and a local machine.

## Project tasks
This project is split into two distinct forecasting problems, since they
differ fundamentally in data volume, methodology, and realistic ambition:

1. **Short-term weather forecasting** (hours to days): a machine
   learning problem, with enough historical hourly data to train real
   models.
2. **Long-term climate trend analysis** (years to decades): a classical
   statistical analysis problem, since yearly-resolution data spans only
   a few decades and is not suited to data-hungry ML approaches.

## 1) Short-Term Weather Forecasting

**Goal:** Predict short-term weather conditions for Germany (e.g. next-day
temperature, rain probability, wind speed) based on current meteorological
conditions in Germany and a set of selected surrounding countries. 

**Approach: two model families are built and compared:**

- **Custom neural network built from scratch**: a configurable
  feed-forward network (variable number of layers and neurons per layer)
  with a custom cost function, implemented without relying on an existing
  ML framework's model classes. The aim is to understand the mechanics
  (forward pass, backpropagation, gradient descent) rather than just use
  a library.
- **Ready-to-use models**: established, freely available implementations
  (e.g. gradient boosting, random forest, or a standard neural network
  library) trained on the same data, used as a benchmark to evaluate
  how the custom implementation compares.
- Both approaches are evaluated against a **naive baseline** (e.g.
  persistence forecast or seasonal average), to ensure any added
  complexity is actually justified by improved accuracy.

## 2) Long-Term Climate Trend Analysis

**Goal:** Analyze long-term climatic development using yearly-aggregated
meteorological data alongside other physical and socio-economic
indicators, using classical statistical methods (trend regression,
correlation analysis) rather than machine learning, given the limited
number of yearly data points available (on the order of decades, not
millions of samples).

## Data Extraction

- **Historic data** is downloaded once via Python scripts and used for both ML
  training and trend analysis. Data covers meteorological
  variables as well as other physical and socio-economic indicators
  relevant to climate development.
- **Current data** (the last hours/days) is downloaded at most once per
  day and used purely as input to already-trained models for inference.
- All downloader scripts include a proper testing structure (mocked
  network calls, retry/error handling) to avoid silent failures or
  corrupted data during automated downloads.

## Parameter Choice (Data Analysis)

- Which variables are downloaded and used for model training is central
  to building a good model. Input parameters are chosen based on
  established meteorological and climatological research on what
  typically correlates with weather and climate development.
- Storing data for every country on the planet is not feasible on a local
  setup. Weather and other selected data sources are therefore reduced
  to the top-N most relevant contributing locations. This selection is
  performed via cross-correlation analysis between candidate countries
  and the target region (Germany) using historical data, rather than
  being chosen by geographic proximity alone.
