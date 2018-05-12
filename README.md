# DeepLearning_H2O_TimeSeries
This is the code associated to the article "*A novel scalable approach based on deep learning for big data time series forecasting*", sent to the **Integrated Computer-Aided Engineering** magazine.

The code is written for the R language. To run it, you must be installed correctly the *h2o, rsparkling, sparklyr and dplyr* packages (you should be carefull with the version of these packages and your R version). Moreover, you must have a cluster of machines with the *Spark and H2O framework* working successfully to run the algorithm in distributed mode.

The input data format to run the code should be a matrix with the following format:

| 1             | 2             | 3       | ...     | w         | w+1        | w+2        | ...        | w+h        |
| ------------- |:-------------:| -----:  | -----:  | -----:    | -----:     | -----:     | -----:     | -----:     |
| data1         | data11        | data12  |  x      | x         | x          | x          | x          | x          |
| data2         | data21        | data22  |   x     | x         | x          | x          | x          | x          |
