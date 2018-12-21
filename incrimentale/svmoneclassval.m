function [pred,ypred]= svmoneclassval(x,xsup,alpha,rho,kernel,kerneloption);

pred=[];
K=normalizekernel(x,kernel,kerneloption,xsup);
ypred=K*alpha+rho;
pred(ypred>-4e-4)=0;
pred(ypred<=-4e-4)=1;