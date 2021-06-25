clear
clc
close all

%% Load data and preparing descriptor labels
load data

x = categorical(desc_name);
x = reordercats(x,desc_name);

%% scaling features
% Range of each descriptor is scaled and mapped between 0 and 1
mi=min(data); 
mx=max(data);
data_sc=(data-mi)./repmat(mx-mi,size(data,1),1);

%% Performing PCA
% The scaled data was presented to PCA. Columns of the score matrix are the
% coordination of samples (molecules) in the reduced space defined by PCA.
% The coeff matrix include the loading values and help to distinguish the
% contribution of each variable (descriptor) to define PC directions. Score
% and load plots can be labeled with docking score and descriptor names
% respectively

[coeff,score,latent,tsquared,explained,mu] = pca(data_sc);
cumvar=cumsum(explained);
figure(1) % Variance information
subplot(1,2,1);bar(latent(1:10));xlabel('PCA Component');ylabel('Eigenvalue');
subplot(1,2,2);bar(cumvar(1:10));xlabel('First N components');ylabel('Percent of accumulative explained variance');

figure(2) % Score plot (2D) 
subplot(1,2,1)
scatter(score(:,1),score(:,2))
text(score(:,1),score(:,2),num2str([1:size(data_sc,1)]'))
axis equal
xlabel(['PC1 (',num2str(explained(1)),'%)'])
ylabel(['PC2 (',num2str(explained(2)),'%)'])

subplot(1,2,2) % Score plot (2D) 
scatter(score(:,2),score(:,3))
text(score(:,2),score(:,3),num2str([1:size(data_sc,1)]'))
axis equal
xlabel(['PC2 (',num2str(explained(1)),'%)'])
ylabel(['PC3 (',num2str(explained(2)),'%)'])

figure(3) %score plot (3D)
scatter3(score(:,1),score(:,2),score(:,3))
text(score(:,1),score(:,2),score(:,3),num2str([1:size(data_sc,1)]'))
axis equal
xlabel(['PC1 (',num2str(explained(1)),'%)'])
ylabel(['PC2 (',num2str(explained(2)),'%)'])
zlabel(['PC3 (',num2str(explained(3)),'%)'])

figure(4) % Loading values
y = coeff(:,1:3);
bar(x,y,'stacked');legend('PC1','PC2','PC3')

figure(5) % Loading plot
scatter3(coeff(:,1),coeff(:,2),coeff(:,3),'.')
hold on
for i=1:size(coeff,1)
    line([0 coeff(i,1)],[0 coeff(i,2)],[0 coeff(i,3)])    
end
text(coeff(:,1),coeff(:,2),coeff(:,3),desc_name)
hold off
axis equal
xlabel('1st PC')
ylabel('2nd PC')
zlabel('3rd PC')

figure(6) % Score plot with docking score value
size_vec=(abs(dock_score_val)-min(abs(dock_score_val)))/(max(abs(dock_score_val))-min(abs(dock_score_val)));
hold on
for i=1:size(score,1)
    plot3(score(i,1),score(i,2),score(i,3),'o','MarkerSize',25*(size_vec(i)+eps),'MarkerFaceColor',[size_vec(i) 0 1-size_vec(i)],'MarkerEdgeColor',[1 1 1])
end
hold off
axis equal
xlabel(['1st PC (',num2str(explained(1)),'%)'])
ylabel(['2nd PC (',num2str(explained(2)),'%)'])
zlabel(['3rd PC (',num2str(explained(3)),'%)'])
title('Score plot')

%% Multiple Linear Regression
% The scaled data matrix and the corresponding docking score values used to
% build MLR model. The regresion coefficient (b) can specify the
% contribution of each variable (descriptor) to model docking score. 

[b,bint,r,rint,stats] = regress(dock_score_val,[data_sc,ones(size(data_sc,1),1)]);
mdl = fitlm(data_sc,dock_score_val);

%% K-means clustring on reduced space
% The reduced data was presented to K-means clustering to find similar
% compounds and patterns in samples (compounds). The optimal number of 
% clusters was 10 and it was determined using Calinski-Harabasz criterion 

[idx,C,sumd,D] = kmeans(score(:,1:3),10);

figure(7) % Score plot labelled based on clusters found by K-means
scatter3(score(:,1),score(:,2),score(:,3),25,idx,'filled')
text(score(:,1),score(:,2),score(:,3),num2str(idx))
axis equal
xlabel('1st PC')
ylabel('2nd PC')
zlabel('3rd PC')

%% Linear Discriminant Analysis  (LDA)
% The range of docking score values is divided in three equal bins and each
% sample assigned to a bin based on the binding score value.
% The LDA loadings can be used to reduced dimensionality of the data and
% simultaneously increase the separability of classes.

bins=linspace(max(dock_score_val),min(dock_score_val),4);
Y3=3*(dock_score_val<bins(3));
Y2=2*and(dock_score_val<bins(2),dock_score_val>bins(3));
Y1=1*(dock_score_val>bins(2));

Y=Y1+Y2+Y3;

Mdl = fitcdiscr(data_sc,Y);
nn=norm(Mdl.Coeffs(1,2).Linear);
bar((Mdl.Coeffs(1,2).Linear)/nn) 

[U,S,V]=svds(inv(Mdl.Sigma)*Mdl.BetweenSigma,2);
D=data_sc*U;
figure(8)
scatter(D(:,1),D(:,2))
text(D(:,1),D(:,2),num2str(Y))

figure(9)
bar(x,U,'stacked');legend('LDA Loading 1','LDA Loading 2')