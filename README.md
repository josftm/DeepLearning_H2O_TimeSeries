# DeepLearning_H2O_TimeSeries
This is the code associated to the article "*A novel scalable approach based on deep learning for big data time series forecasting*", sent to the **Integrated Computer-Aided Engineering** magazine.

The code is written for the R language. To run it, you must be installed correctly the *h2o, rsparkling, sparklyr and dplyr* packages (you should be carefull with the version of these packages and your R version). Moreover, you must have a cluster of machines with the *Spark and H2O framework* working successfully to run the algorithm in distributed mode.

The input data format to run the code should be a matrix with the following format:

|               		|        		        |         						|       					|       |        				    |
| ------------- 		| ------------- 		| ----- | -----  				| -----   					| ----- | -----     				|
| X<sub>1</sub>			| X<sub>2</sub>			| ... 	| X<sub>w</sub>			| X<sub>w+1</sub>			| ...	| X<sub>w+h</sub>         	|
| X<sub>1+h</sub>		| X<sub>2+h</sub>		| ... 	| X<sub>w+h</sub>		| X<sub>w+h+1</sub>			| ...	| X<sub>w+2h</sub>        	|
| X<sub>1+2h</sub>		| X<sub>2+2h</sub>		| ... 	| X<sub>w+2h</sub>		| X<sub>w+2h+1</sub>		| ...	| X<sub>w+3h</sub>        	|
| X<sub>1+3h</sub>		| X<sub>2+3h</sub>		| ... 	| X<sub>w+3h</sub>		| X<sub>w+3h+1</sub>		| ...	| X<sub>w+4h</sub>        	|
