# Project Rossmann - Sales Predict with Machine Learning

### This project is an application of machine learning models to predict sales based on public pharmaceutical stores data from Rossman provided by Kaggle. The project is not intended to propose solutions for the company, being only an object of demonstration of analytical skills and machine learning.
- Dataset Link: https://www.kaggle.com/competitions/rossmann-store-sales/data

<p align="center">
  <img src="images/ross.png">
</p>


## Understanding Business Questions 

Rossmann operates over 3,000 drug stores in 7 European countries. Currently,
Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors,
including promotions, competition, school and state holidays, seasonality, and locality.
With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.


- At a meeting with the leads of each department, the Rossmann's CFO made a proposal to renovate all of their store.

- The Rossmann's CFO wants to predict the next 6 weeks sales for each store, so he can bring forward part of this revenue to renovate them.

- The CFO, who directly requested the answer for the problem.

- Granularity: daily sales for 6 weeks / store.

- Problem type: sales prediction.


## Solution Planning

1. An ovewview at data description and create Hypothesis to analyse realtionship between variables to help in data selection for machine learning model.
2. Create a mindmap to understand features and build hypothesis.

<p align="center">
    <img src="images/mindmap.jpg" width="900" height="600">
</p>

3. Feature engineering to help analyze the hypotheses and add some more informations.
4. Exploratory data analysis to understand distribuition, scale, nature, correlations, etc...
5. Data preparation to tranform nature, encoding and rescaling data.
6. Feature selection to help ML model .
7. Applying ML models for sales prediction and compare which one has better error measures.
8. Hyperparemeter Fine Tuning to improve the performance of the selected model.
9. Error interpretation, make results easier for non-technical people.
10. Deploy model to production


## Main Insights

#### Hypothesis creation and correlation visualization.

- H1: Stores with the highest assortment are expected sell more, on average
  - **TRUE - Stores that have extra assortment sell 18% more than extended and, extended assortment sells 10% more than basic assortment**

- H2: Stores in promotion are expected to sell more, on average
  - **TRUE - Stores that have promotion sell 10% more**
  
- H3: Stores with close competitors should sell less, on average.
  - **FALSE - The distance from the competitor does not significantly influence sales**
  
- H4: Stores should sell more in holidays, on average
  - **TRUE - The average sales are higher on holidays than regular days, regular days have the lowest average sales**

- H5: Christmas is the best selling holiday
  - **FALSE - Easter holiday is the holiday that has the highest sales**

- H6: Stores with more time of promotion should sell more
  - **FALSE - Stores with more consecutive promotions sell less**

- H7: Stores with longer competitors are expected to sell more.
  - **FALSE - Stores that have competitior that have recently opened sell more, there is a drop in sales over time**
  
- H8: Stores sells more in the second semester, on average
  - **TRUE - Stores sell 2.7% more in the second semester**

- H9: Stores should sell more over the years 
  - **TRUE - Stores sell about 2% more over year, on average**

- H10: Stores sell more on weekends
  - **FALSE - Stores sell less on weekends**



#### Hypotheses and results:

|Hypothesis  |  Conclusion  |  Relevance  |
|----------- | -----------  | ------------|
|H1          | True         | Low         |
|H2          | True         | Medium      |
|H3          | False        | High        |
|H4          | True         | low         |
|H5          | False        | Medium      |
|H6          | False        | High        |
|H7          | False        | High        |
|H8          | True         | Medium      |
|H9          | True         | Low         |
|H10         | False        | Medium      |


## Correlation

#### Numeric - Pearson Correlation
<p align="center">
  <img src="images/corr.png"/>
</p>

## Results

* Answering questions
  - The purchase suggestion was created following the solution assumptions.
  - 4025 homes were suggested for purchase
  - The total profit is $401,706,161.00 
  - The profit percentage is 23.60%
  - Best time to sell is spring and summer or first semester of year
  - To access remotely, a dashboard was created and deployed using heroku 

## Deploy

- Link: https://analytics-kc-house.herokuapp.com/

<p align="center">
  <img src="images/i3.jpeg"/>
</p>

<p align="center">
  <img src="images/i4.jpeg"/>
</p>

<p align="center">
  <img src="images/i1.jpeg"/>
</p>

<p align="center">
  <img src="images/i2.jpeg"/>
</p>

