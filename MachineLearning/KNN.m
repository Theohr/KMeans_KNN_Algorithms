 clear; close all; clc;

 YDataset = gen_superdata(10578755);
   
 [RowsYDataset,ColumnsYDataset] = size(YDataset); % produces the rows and columns of the dataset
 
 %displays the rows
 RowsYDataset
 
 MeanYDataset = mean(YDataset); % produces the mean of each column in the dataset
 
 StandDevYDataset = std(YDataset); % produces the standard deviation of each column in the dataset
 
 % Displays both mean and standard dev
 MeanYDataset
 StandDevYDataset
 
 % produces and displays the covariance matrix of the dataset
 CovarYDataset = cov(YDataset(:,1:5))
 
 % procudes and displays the correlation matrix of the dataset
 CorrYDataset = corrcov(CovarYDataset)
  
 % gets all all the values of the last column
 Classes = YDataset(:, end);
 
 % produces the classes of the dataset
 ClassesAns = unique(Classes)
 
 DatasetSize = size(YDataset,1); % gets the rows number of the dataset for the training testing
 Samples = DatasetSize*0.6; % gets 60% of the samples of the dataset
 randomRows = randperm(DatasetSize); % randomizes the rows 
     
 TrainingYDataset = YDataset(randomRows(1:Samples),:); % produces the training dataset
 TestingYDataset = YDataset(randomRows(Samples:end),:); % produces the testing dataset
 
 TrainingYDataset2 = TrainingYDataset(:,1:5); % produces the training dataset without the 6th column
 TestingYDataset2 = TestingYDataset(:,1:5); % produces the testing dataset without the 6th column
 ClassTraining = TrainingYDataset(:,6); % produces the class training with the 6th column onlu
 ClassTesting = TestingYDataset(:,6); % produces the class testing with the 6th column only
 
 % creates a loop which will check the k values for 5 and 7
 for n=1:2
     
 k = n * 2 + 3; % first time k will equal 5 then 7
 Mdl = fitcknn(TrainingYDataset2,ClassTraining,'NumNeighbors',k) % training the KNN classifier

 %creates a for loop using predict feature to predict KNN testing examples
 for i=1:size(TestingYDataset2,1)
   TestingExamples = TestingYDataset2(i,:); % 
   PredictingKNN(i) = predict(Mdl,TestingExamples);
 end
 
 % Confusion matrix for testing data
 for i=1:6   % 1 to 6 which is the number of classes
  index=find(ClassTesting==i);  % index of all data points as per classes
  number=length(index);                 %number of index per classes
  
  for j=1:6
    Classification=length(find(PredictingKNN(index)==j)); % Correct classification for each class 
    ConfusionMatrix(j,i)=Classification/number*100;   % calculate the percentage 
  end
 end
 
   display (k) % displays the k  
   display (ConfusionMatrix,'Confusion matrix for testing data '); % displays the confusion matrix

   % calculating the average correct classification
   AverageCorrectClassification=length(find((PredictingKNN-ClassTesting')==0))/length(ClassTesting)*100; 
    
    display (AverageCorrectClassification, 'Percentage of correct classifications for testing data ')
    %displaying the average correct classification
 end

