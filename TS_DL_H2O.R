"
Set max memory and load required libraries. These libraries are:
- h2o: Open source math engine for big data that computes parallel distributed machine learning algorithms.
- sparklyr: Allow connect to Spark from R.
- rsparkling: Extension package for sparklyr that creates an R front-end for the Sparkling Water package from H2O.
- dplyr: Provides a set of tools for efficiently manipulating datasets in R.
"
options( java.parameters = "-Xmx16g" )
library(h2o)
library(rsparkling)
library(sparklyr)
library(dplyr)

"
Spark installation and set the sparklingwater version.
Note: Both versions must be the same.
"
spark_install(version = "2.0.2")
options(rsparkling.sparklingwater.version = "2.0.2")

"
Setting variables:
- w: The historical windows size.
- h: The prediction horizon.
- dir: Directory containing the data set.
"
w = 168
h = 24
dir = "YOUR_WORKING_PATH_ONLY"

"
Create a Spark conection into a local installation using all available threads, 2.0.2 version and setting the max memory of executor (*) and driver to 16 GB.
If you want to launch into a cluster, change local[*] by your Spark cluster URI (i.e. spark://IP_Master:7077).
Furthermore, you can limit the hardware resources using sparklyr config parameters, as spark.executor.cores to limit the number of cores.
"
sc <- spark_connect(master = "local[*]", version = "2.0.2",
                    config = list("sparklyr.shell.executor-memory"="16g", 
                                  "sparklyr.shell.driver-memory"="16g"))

"
Generate the h2oContext using the spark connection
"
h2o_context(spark_connection(sc))

"
Load the dataset from dir variable into Spark a in-memory spark dataframe.
Then, cast it to a distributed h2oFrame using the spark connection.
This provide the in-memory dataset fully distributed.
"
data_df <- spark_read_csv(sc, "dataset", sprintf("%s/DATA_FILE_NAME.csv",dir), header=FALSE, delimiter = " ")
data_hf <- as_h2o_frame(sc, data_df)

"
Split the data into training and test set.
After that, split the training set into training and validation set in order to search the optimal parameters.
"
split = 0.7
corte = floor(split*nrow(data_hf))
data.training = data_hf[1:corte,]
data.test = data_hf[(corte+1):nrow(data_hf),]
corte.val = floor(split*nrow(data.training))
data.traininVal = data.training[1:corte.val,]
data.validation = data.training[(corte.val+1):nrow(data.training),]

"
Hyper-parameters definition:
- predictors: subset of variables that will be used to train the model. This variable corresponds to the entire historical window (w).
- neurons: initialisation of the number of hidden layer and neurons per layer. In this example, the number of hidden layers goes from 1 to 5. The number of neurons per layers are from 10 to 100 into a 10 value interval. 
- lambda: regularization parameter.
- activation: The activation function (non-linearity) to be used the neurons.
- distribution: The distribution from which initial weights are to be drawn.
- stop.metric: The stopping criteria in terms of regression error on the training data scoring dataset.
"
predictors = c(1:w)
neurons = list(10,20, 30,40,50,60,70,80,90, 100,
               c(10,10),c(20,20), c(30,30),c(40,40), c(50,50),c(60,60), c(70,70),c(80,80),c(90,90), c(100,100),
               c(10,10,10),c(20,20,20), c(30,30,30),c(40,40,40), c(50,50,50),c(60,60,60), c(70,70,70),c(80,80,80),c(90,90,90), c(100,100,100),
               c(10,10,10,10),c(20,20,20,20), c(30,30,30,30),c(40,40,40,40), c(50,50,50,50),c(60,60,60,60), c(70,70,70,70),c(80,80,80,80),c(90,90,90,90), c(100,100,100,100),
               c(10,10,10,10,10),c(20,20,20,20,20), c(30,30,30,30,30),c(40,40,40,40,40), c(50,50,50,50,50),c(60,60,60,60,60), c(70,70,70,70,70),c(80,80,80,80,80),c(90,90,90,90,90), c(100,100,100,100,100)
)
lambda = c(0.01, 0.0001)
activation = "Tanh"
distribution = c("gaussian","poisson")
stop.metric = "MSE"
hyper_params <- list(hidden = neurons, 
                     l1 = lambda, 
                     activation =activation,
                     distribution = distribution,
                     stopping_metric=stop.metric
)

"Create output directory"
path = sprintf("%s/Lambda%s_%s",dir,lambda,distribution)
dir.create(path)

"Create output variables"
info = list()
errors = NULL
best = vector()
predictions = vector()
pre.agreg = NULL
subpr.valError = vector()

"Times variables"
models.time = 0
pred.time = 0

"
For each subproblem:
- Set the target index to forecast.
- Compute a grid with all possible combinations specified in hyper_params variable.
- Retrieve the best model.
- Get the metrics
"
for(i in 1:h){
  y <- w+i 
  time.start.model = proc.time()
  model_grid = h2o.grid("deeplearning",
                        hyper_params = hyper_params,
                        x = predictors,
                        y = y,
                        training_frame = data.traininVal,
                        validation_frame = data.validation)
  
  #Retrieve the best model of the grid (first position of the list)
  best_model = h2o.getModel(model_grid@model_ids[[1]])  
  time.stop.model = proc.time()
  
  #Append the rmsle in % from the validation set
  subpr.valError = append(subpr.valError, best_model@model$validation_metrics@metrics$rmsle * 100)
  
  #Get execution time
  models.time = models.time+(time.stop.model[3]-time.start.model[3])
  
  #Append de best model to model list
  best = append(best, best_model) 
  
  #Predict the test set using the best model obtained and get the execution time of the prediction
  data.w = data.test
  time.start.predict = proc.time()
  pre = h2o.predict(object = best_model, newdata = data.w)
  time.stop.predict = proc.time()
  pred.time = pred.time+(time.stop.predict[3]-time.start.predict[3])
  predictions = append(predictions,pre)
  
  "
  Get metrics and errors and append into two tables:
  - pre.agreg: A matrix with all predictions (each column corresponds to a subproblem)
  - errors: contains all the metrics for each subproblem
  "
  mse = 1/nrow(pre)*sum((data.test[,y]-pre)^2)
  rmse = sqrt(1/nrow(pre)*sum((data.test[,y]-pre)^2))
  mae = 1/nrow(pre)*sum(abs((data.test[,y]-pre))^2)
  rel = ( 1 / nrow(pre)) * sum( abs(( data.test[,y] - pre )) / data.test[,y]) * 100
  pre.agreg = cbind(pre.agreg, as.vector(pre))
  errors = rbind(errors, c(y,mse, rmse, mae, rel))
  
  #Delete the grid and the best model in order to repeat the proccess with other subproblem
  rm(model_grid)
  rm(best_model)
}

colnames(errors) = c("H","MSE","RMSE", "MAE", "RELAT")

#Save the best model for each subproblem into a file. This includes the optimal parameters configuration.
cat(NULL,file=sprintf("%s/models.txt",path))
sink(sprintf("%s/models.txt",path)) 
print(best)
sink()

#Save a table with the comparison between real and predicted into a file (2 column matrix).
class_test = as.matrix(data.test[,(w+1):ncol(data.test)])
results = as.matrix(as.vector(t(class_test)))
results = cbind(results, as.vector(t(pre.agreg)))
colnames(results) <- c("REAL","PREDICTION")
write.table(results, file=sprintf("%s/comparison.csv",path), row.names=FALSE, col.names = TRUE, sep = " ")

#Compute the acumulative metrics. This means the metrics for all subproblem
err.accum = NULL
mse_acum = 1/(nrow(pre)*ncol(pre))*sum(sum((class_test-pre.agreg)^2))
rmse_acum = sqrt(1/(nrow(pre)*ncol(pre))*sum(sum((class_test-pre.agreg)^2)))
mae_acum = 1/(nrow(pre)*ncol(pre))*sum(sum(abs((class_test-pre.agreg))^2))
rel_acum = 1 / (nrow(pre.agreg)*ncol(pre.agreg)) * sum(sum(abs(pre.agreg - class_test ) / class_test)) * 100
err.accum = rbind(err.accum, c(mse_acum, rmse_acum, mae_acum, rel_acum))
colnames(err.accum) = c("MSE","RMSE", "MAE", "RELAT")

#Save the prediction for each subproblem as matrix form into a file
write.table(pre.agreg, file=sprintf("%s/prediccion_as_matrix.csv",path), row.names=FALSE, col.names = FALSE, sep = " ")

#Save all configurations and another interesting info into a file
info["MEDIA_RMSLE_VALIDACION"] = mean(subpr.valError)
info["NEURONAS"] = list(neurons)
info["LAMBDA"] = lambda
info["ACTIVATION"] = activation
info["DISTRIBUTION"] = distribution
info["STOPPING_METRIC"] = stop.metric
info["ERRORS_PER_SUBPROBLEM"] = list(errors)
info["GLOBAL_ERRORS"] = list(err.accum)
info["GET_MODEL_TIME"] =  models.time 
info["PREDICTION_TIME"] =  pred.time
sink(sprintf("%s/results.txt",path)) 
print(info)
sink()

"Disconect from Apache Spark"
spark_disconnect_all()