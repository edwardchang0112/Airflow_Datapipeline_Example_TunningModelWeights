# Airflow_Datapipeline_Example_TunningModelWeights
In this project, there is an example for you to build you tasks on airflow, such as model trainning, testing performance, etc.

## As an example for every one can easily deploy projects on airflow

For this project, we try to use amount of user's data on using our product, which is skincare related, to train a model, then based on the model to recommand user a customized solution. Hence, tunning a better weights for the model is important.

## Setting

1. Install airflow
2. $airflow start
3. $airflow webserver -p 8080 (the ip you want)
4. $airflow scheduler
5. Go to localhost:8080, then there it is!

## Tasks

This project try to do following tasks:

1. Load_data: Load part of user's data on using the product. (you can change data you want)
2. Loop: Try to fit the model with users' data, and tune the weights (in loop) based on the cross validation result.
3. Store_weights: Store the best weights from last step for future use.

### The data pipeline figure on airflow

This provide you the basic structure, and you can follow this example to extends the data pipeline to extend the stucture for your application.

![image]()
