%%
close all
clear all
clc

%%
%1)
load ('dd_tools/gamma1.mat')
figure(1)
subplot(1,3,1)
plot(c1(:,1),c1(:,2),'r*',c2(:,1),c2(:,2),'g*')
title('Résultat de classifier SVDD incrémental ')
%%
%2)
% Tenant compte de l’allure de la courbe obtenue, nous optons pour le cas
% %Gaussian donc nous choisissons comme kernel rbf
%%
%3)
[train1,test1,val1] = divideblock(c1.', .66,.16, .17);
test1 = test1.';
val1 = val1.';
xtest = gendatoc(test1,val1) ;
xtrain = train1.'; 
 
[train2,test2,val2] = divideblock(c2.', .66,.16, .17);
test2 = test2.';
val2 = val2.';
xtest2 = gendatoc(test2,val2) ;
xtrain2 = train2.';
 
train = gendatoc(xtrain,[]) ;
test = gendatoc(xtest,c2);
%% 
%4)
tic
W = incsvdd([],0.2,'r',19);
Wtrain = train * W ; 
temps= toc;
plotc(Wtrain,'b--')
%%
%5) 
roc = dd_roc(test,Wtrain);
auc = dd_auc(roc);
msgbox(['accuracy = ',num2str(auc)])
msgbox(['Execution time = ',num2str(temps)])
%%
%7)
subplot(1,3,2)
plotroc(roc,'r')
title(' Roc Curve ')
labels = labelss(test*Wtrain*labeld).';
a = ones(1,length(xtest));
b = zeros(1,length(test)-length(xtest));
t_labels = [a b].';
conf = confusionmat(t_labels,labels).';
%%
subplot(1,3,3)
plotConfMat(conf,{'c2','c1'},test,Wtrain)
title('Confusion matrix')
figure(7)
subplot(1,4,1)
plotConfMat(conf,{'c2','c1'},test,Wtrain)
title('Confusion matrix pour iSVDD')

%%
%Q8 
figure(2)
subplot(1,3,1)
plot(c1(:,1),c1(:,2),'r*',c2(:,1),c2(:,2),'g*')
title('Résultat de classifier  one-class  SVM  incrémental ')
train2 = gendatoc(xtrain,xtrain2) ;
tic;
W1 = incsvc([],'r',1,19);
Wtrain2 = train2 * W1 ; 
temps2 = toc;
plotc(Wtrain2,'b--')
%%
%9)
roc2 = dd_roc(test,Wtrain2);
subplot(1,3,2)
auc2 = dd_auc(roc2);
plotroc(roc2,'b')
title(' Roc Curve ')
msgbox(['accuracy = ',num2str(auc2)])
msgbox(['Execution time = ',num2str(temps2)])
labels = labelss(test*Wtrain2*labeld).';
conf = confusionmat(t_labels,labels).';
%%
subplot(1,3,3)
plotConfMat(conf,{'c2','c1'},test,Wtrain2)
title('Confusion matrix')
figure(7)
subplot(1,4,2)
plotConfMat(conf,{'c2','c1'},test,Wtrain2)
title('Confusion matrix pour iOSVM (batch)')
%%
%10) 
% Nous constatons que la valeur de l’AUC2 est supérieure à celle de l’AUC1. % De même pour le ROC.
%%
%11)
figure(3)
subplot(1,3,1)
plot(c1(:,1),c1(:,2),'r*',c2(:,1),c2(:,2),'g*')
title('Résultat de classifier  SVDD (batch)')
tic
W3 = svdd([],0.2,19); 
Wtrain3 = train * W3;
temps3 = toc; 
plotc(Wtrain3,'b--')
roc3 = dd_roc(test,Wtrain3);
auc3 = dd_auc(roc3);
subplot(1,3,2)
plotroc(roc3,'k')
title(' Roc Curve ')
labels = labelss(test*Wtrain3*labeld).';
conf = confusionmat(t_labels,labels).';
msgbox(['accuracy = ',num2str(auc3)])
msgbox(['Execution time = ',num2str(temps3)])
%%
subplot(1,3,3)
plotConfMat(conf,{'c2','c1'},test,Wtrain3)
title('Confusion matrix')
figure(7)
subplot(1,4,3)
plotConfMat(conf,{'c2','c1'},test,Wtrain3)
title('Confusion matrix pour SVDD (batch)')
%%
tic
[xsup,alpha,rho,pos] = svmoneclass(train2,'htrbf',[1,1.5],0.2,0); 
temps4 = toc; 
[labels,ypred] = svmoneclassval(test,xsup,alpha,rho,'htrbf',[1,1.5]);
conf = confusionmat(t_labels,labels).';
msgbox(['Accuracy = ', num2str(100*trace(conf)/sum(conf(:))),' %']);
msgbox(['Execution time = ',num2str(temps4)])
auc4=100*trace(conf)/sum(conf(:));

%%
figure(4)
plotConfMat(conf,{'c2','c1'},test,Wtrain3)
title('Confusion matrix pour one-class SVM')
figure(7)
subplot(1,4,4)
plotConfMat(conf,{'c2','c1'},test,Wtrain3)
title('Confusion matrix pour OSVM')
%%
figure(5)
plotroc(roc,'r')
plotroc(roc2,'b')
plotroc(roc3,'k')
legend('iSVDD','','iOSVM','','SVDD (batch)','');
title(' Roc Curve ')
%%
figure(6)
plot(c1(:,1),c1(:,2),'y*',c2(:,1),c2(:,2),'g*')
plotc(Wtrain,'r--')
plotc(Wtrain2,'b--')
plotc(Wtrain3,'k--')
legend('C1','C2','iSVDD','iOSVM','SVDD (batch)');
%%

% plotting the colors
confpercent=[auc*100,auc2*100,auc3*100,auc4];
imagesc(confpercent);
title('Accuracy for each model');
% set the colormap
colormap(parula(4));

% Create strings from the matrix values and remove spaces
textStrings = num2str(confpercent(:));
textStrings = strtrim(cellstr(textStrings));
textStrings(1)=strtrim(cellstr(strcat('iSVDD :  ',textStrings(1),' %')));
textStrings(2)=strtrim(cellstr(strcat('iOSVM :  ',textStrings(2),' %')));
textStrings(3)=strtrim(cellstr(strcat('SVDD(batch) :  ',textStrings(3),' %')));
textStrings(4)=strtrim(cellstr(strcat('OSVM :  ',textStrings(4),' %')));


% Create x and y coordinates for the strings and plot them
[x] = 1:4;
[y] = ones(1,4);
hStrings = text(x(:),y(:),textStrings(:), ...
    'HorizontalAlignment','center');

% Get the middle value of the color range
midValue = mean(get(gca,'CLim'));

% Choose white or black for the text color of the strings so
% they can be easily seen over the background color
textColors = zeros(4,4);%repmat(confpercent(:) > midValue,1,3);


set(hStrings,{'Color'},num2cell(textColors,2));