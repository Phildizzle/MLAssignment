Disclaimer
==========

This repository contains the final programming assignment for the
practical machine learning course in the Data Science specialization of
JHU on Coursera.

Executive Summary
=================

Thanks to new fitness devices such as *Jawbone Up, Nike FuelBand,* and
*Fitbit* it is more easy to collect a large amount of measurement about
personal activity. This data is used by quantitatively oriented fitness
geeks to steadily improve their physical performance. One thing that
people regularly measure is **how much or long** of an activity they
perform, however they rarely quantify \*\* how well\*\* they do it. In
this project our goal is to use data of accelerometers on the belt,
glove, upper arm, and dumbell to predict if an exercise is correctly
performed or not. Further, we assess the feasibility of automatically
assessing the quality of execution of weight lifting exerises. We
gratefully acknowledge the provision of the data by Velloso et al.,
2013, see <http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201>).

Six young men (aged 20-28) were asked to perform one set of 10
repetitions. Each repetition counts as one observation. Participants
were asked to perform one set of 10 repetitions of the Unilateral
Dumbbell Biceps Curl in five different fash- ions: exactly according to
the specification (Class A), throw- ing the elbows to the front (Class
B), lifting the dumbbell only halfway (Class C), lowering the dumbbell
only halfway (Class D) and throwing the hips to the front (Class E).
Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common lifting mistakes.

After an extensive cleaning of the data set. We test our hypothesis by
emplyoing machine learning techniques, i.e. we strive to find accurate
predictions if an exercise was done correctly or not. We test three
different algorhythms with machine learning: a random forest model
("rf"), boosting with trees ("gbm"), and linear discriminant analysis
("lda"). The random forest model has the best accuracy which is why we
use it for predicting the test data set.

Load and explore the data
=========================

Let's start by loading and exploring the data set.

    # Change the working directory (not really necessary) 
    setwd("C:\\Users\\user1\\Desktop\\Data Science\\8. Machine Learning\\Programming Assignment")

    # Load packages
    library(caret)
    library(dplyr)
    library(parallel)
    library(doParallel)

    # Download and load Files.
    train.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(train.url, destfile="pml-training.csv")
    test.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(test.url, destfile="pml-testing.csv")
    WLTraining <- read.csv("pml-training.csv")
    WLTesting <- read.csv("pml-testing.csv")
    # Do not get confused "WLTraining" refers to the whole data set we use for the machine learning exercise.
    # "WLTesting" refers to the data set we need for the quiz.

    # Let us check how large our data sets are:
    dim(WLTraining)

    ## [1] 19622   160

    dim(WLTesting)

    ## [1]  20 160

Let us split the data set into a "training" and "testing" set so we can
proceed.

    set.seed(0-100) # Cause our analysis goes from 0-100 yo!

    trainid <- createDataPartition(WLTraining$classe, p=0.7, list=F)

    Training <- WLTraining[trainid,]
    Testing  <- WLTraining[-trainid,]

    # Next we will save our outcome to be predicted in another variable
    train.classe <- WLTraining[trainid, "classe"]
    test.classe  <- WLTraining[-trainid, "classe"]

    # Finaly our data can be explored: 
    dim(Training)

    ## [1] 13737   160

    dim(Testing)

    ## [1] 5885  160

    str(Training[,1:10])

    ## 'data.frame':    13737 obs. of  10 variables:
    ##  $ X                   : int  1 2 3 5 6 7 8 10 11 12 ...
    ##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2: int  788290 808298 820366 196328 304277 368296 440390 484434 500302 528316 ...
    ##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.45 1.42 1.42 1.45 1.45 1.43 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.07 8.06 8.09 8.13 8.17 8.18 8.18 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...

    ggplot(WLTraining, aes(x=classe, fill=classe)) + geom_bar() + ggtitle("Frequency of classes") + labs(title="Frequency of classes", xlab="Classes", ylab=" Absolute Frequency")

![](Programming_Assignment_files/figure-markdown_strict/unnamed-chunk-2-1.png)
This plot shows the absolute frequencies of different classes.

Taking a closer look at the data set, we find three mentionable points:
1. The data set contains variables not relevant for our analysis. That
is, the first seven columns contain information on subjects. The last
column represents the outcome "classe". 2. Many variables have been
coded as factors whereas they are obviously numerical, see for instance
the column "skewness\_roll\_dumbbell". 3. The data set includes a lot of
NA values.

The first two problems can be easily alleviated by the following code:

    Training <- as.data.frame(apply(Training[,-c(1:7, 160)], 2, as.numeric))
    Testing  <- as.data.frame(apply(Testing[,-c(1:7, 160)],  2, as.numeric))

We exclude columns 1-7 and 160 from the data set and turn all variables
into numeric ones. The latter is obviously a shortcut which in fact
could introduce a bias to our analysis. Yet, the number of factors in
our sample is small. Overall, I think this will not introduce a large
problem.

The third point made is a bit trickier to solve. Let us check how much
of a problem NA values are by finding out how many columns have more
than 95% of missing values:

    NAdata <- colMeans(is.na(Training))
    NAdata <- NAdata[NAdata>.95]
    length(NAdata)

    ## [1] 67

That is quite a large number. However, those columns have little to
contribute we will exclude them from our data set.

    Training <- dplyr::select(Training, which(round(apply(is.na(Training), 2, sum ) / dim(Training)[1], 2) < .95) )
    dim(Training)

    Testing <- dplyr::select(Testing, which(round(apply(is.na(Testing), 2, sum ) / dim(Testing)[1], 2) < .95) )

That looks quite cleaned up. As a nice side effect the exclusion of
those columns will make our machine learing algorhythm go faster later
on! As a last check we are going to use the "nearZeroVar" function to
check if there are zero or near zero covariates

    nearZeroVar(Training)

    ##  [1]  6 12 13 14 15 16 17 18 19 20 43 44 45 46 47 48 52 53 54 55 56 57 58
    ## [24] 59 60 74 75 76 77 78 79 80 81 82

That is basically it vis-à-vis the cleaning, as a next step we will
finally build our machine learning model.

Machine learning model
----------------------

Now before we start, I want to prepare my setup such that our analysis
runs fast as possible. This guide by
([link](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md))
is very heplful in this instance. Some machine learning models, in
particular random forests, may take up lots of your computing power.
Hence, it makes sense to make use of more than one core (R-default). The
following code allows us to do this:

    cluster <- makeCluster(detectCores()-1) 
    # -1 core to let the OS run on it.
    registerDoParallel(cluster)

    fitControl <- trainControl(method = "cv",
                               number = 5,
                               allowParallel = TRUE)

We can finally estimate our machine learning models. We run four models:
random forests, boosting with trees, and linear discrimant analysis:

    model_1 <- train(x=Training, y=train.classe, method = "rf", trControl = fitControl, verbose = FALSE)
    model_2 <- train(x=Training, y=train.classe, method = "gbm", trControl = fitControl, verbose = FALSE)
    model_3 <- train(x=Training, y=train.classe, method = "lda", trControl = fitControl, verbose = FALSE)

    # Do not forget to close to stop the cluster we set up:
    stopCluster(cluster)
    registerDoSEQ()

    # Create predictions
    prediction_1 <- predict(model_1, Testing)
    prediction_2 <- predict(model_2, Testing) 
    prediction_3 <- predict(model_3, Testing)

    # Create prediction matrices
    conf_matrix_1 <- confusionMatrix(prediction_1, test.classe)
    conf_matrix_2 <- confusionMatrix(prediction_2, test.classe)
    conf_matrix_3 <- confusionMatrix(prediction_3, test.classe)

    # Column 8 is the model accuracy
    conf_matrix_1$table; conf_matrix_1$overall
    conf_matrix_2$table; conf_matrix_2$overall
    conf_matrix_3$table; conf_matrix_3$overall

Looks like the random forest model has the highest accuracy. We need
this because we want to predict 20 out of 20 questions correct. The
chance of doing that with an accuracy of 0.99 is 81.8%. Let us predict
the test set.

Predictions
===========

The final step is to predict the test data set comprising of 20
observations, as proposed by the project's specification. We will use
the random forest model above, since it gave the best predictions.

    # Let us prepare the test data set for the prediction by applying the same 
    WLTesting <- as.data.frame(apply(WLTesting[,-c(1:7, 160)],  2, as.numeric))
    WLTesting <- dplyr::select(WLTesting, which(round(apply(is.na(WLTesting), 2, sum ) / dim(WLTesting)[1], 2) < .95) )

    pred_1 <- predict(model_1,WLTesting)
    pred_1

For this particular exercise, the random forest model ("rf")
outperformed both the boosting with trees ("gbm") and linear
discriminant analysis ("lda") models. The random forest model delivers
overall 20 accurate predictions. Nevertheless, the gbm model performed
close to the random forest, suggesting that it could be applicable to a
real world scenario.

######### 

References
==========

1.  Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
    Qualitative Activity Recognition of Weight Lifting Exercises.
    Proceedings of 4th International Conference in Cooperation with
    SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
