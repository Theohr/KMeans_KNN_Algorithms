clear; close all; clc; 

Dataset=gen_kmeansdata(10578755); % get the dataset from the file

[RowsX,ColumnsX] = size(Dataset); %displays the rows and columns of the dataset

 RowsX
 
 MeanDataset = mean(Dataset); % calculates the mean of the dataset
 
 StandDevDataset = std(Dataset); % calculates the standard deviation of the dataset
 
 MeanDataset
 StandDevDataset
 
 % creates a figure for the histogram display for each column
 figure();
 for ColumnsX = 1:4
     subplot(2,2, ColumnsX); %for each column creates a subplot to display the histogram
     hist(Dataset(:,ColumnsX));
     title(['Histogram for column: ',num2str(ColumnsX) ])
 end
 
 % calculates the covariance matrix of the Dataset
 CovarDataset = cov(Dataset)
 
 % calculates the correlation matrix of the dataset
 CorrDataset = corrcov(CovarDataset)
 
 % creates a figure displaying the coordinates of centres of k classes 
 %silhouette plot
 % the mean silhouette measure
 figure('Name', 'Clusters');
 i=1
 for k = 3:5 %Getting the k number
     [index, centroids] = kmeans(Dataset, k); % Calculates the group of points index and the centroid points
     subplot(1,3,i); % Splitting the figure into 3 subplots
     silhouette(Dataset, index); % produces the silhouette plot
     title(['K: ',num2str(k)])
     SilhouetteValue(:,i) = silhouette(Dataset, index); %Produces the silhouette value
     SilhouetteMean(i) = mean(SilhouetteValue(:,i)); %Produces the mean sillhouete value
     i = i + 1;
     centroids
 end
 
 SilhouetteMean
 
 highestSil = max(SilhouetteMean());   %The highest mean in the array
 
 OptNoK = 4 %the optimal value of K is row 4
 
 [indexbest, centroidbest] = kmeans(Dataset, OptNoK); % gives you the best data on average for each cluster
 
 % creates a figure with the 3d projection of clustered data 
 figure('Name','Visualization');
 % gets 3 columns for each cluster and the centroid based on the average best data for each
 % cluster to create the 3d projection 
 plot3(Dataset(indexbest==1,1),Dataset(indexbest==1,2),Dataset(indexbest==1,3),'r.','MarkerSize',12);
 hold on
 plot3(Dataset(indexbest==2,1),Dataset(indexbest==2,2),Dataset(indexbest==2,3),'b.','MarkerSize',12);
 plot3(Dataset(indexbest==3,1),Dataset(indexbest==3,2),Dataset(indexbest==3,3),'g.','MarkerSize',12);
 plot3(Dataset(indexbest==4,1),Dataset(indexbest==4,2),Dataset(indexbest==4,3),'y.','MarkerSize',12);
 plot3(centroidbest(:,1), centroidbest(:,2), centroidbest(:,3), 'kx', 'MarkerSize', 12, 'LineWidth', 3);
 legend('First Cluster', 'Second Cluster', 'Third Cluster', 'Fourth Cluster', 'Centroid', 'Location', 'NE');
 hold off
 
 
