## Copyright (C) 2015 Calus Peng
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%%@ kernel SVM (Support Vector Machine)model implementation for classfication  @%
%%@ problem.		                                                       @%
%%@ this source code is only used for academic  purpose,if you want to use this@% 
%%@ in you academic project,please cite the source <calus peng ,py79hn@163.com>@%
%%@ or if you want to use it for business purpose, please first send a email to@%
%%@ the author for requesting  permission,or you'll be responsible for your    @%
%%@ Illegal action.                                                            @%
%%@                                                                            @%
%%Author: Redmaple <py79hn@163.com>                                            @%
%%Created: 2015-12-02                                                          @%
%%code can be downloaded from github:https://github.com/py79hn/Machine-Learning@%
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%generate the training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data,label] = generateData()
    x0 = 3;
    y0 = 3;
    r1 = 2;
    r2 = 3;
    dd = 0.2;
    [X1,X2] = meshgrid((r2-x0):dd:(r2+x0),(y0-r2):dd:(y0+r2));
    X = [reshape(X1,1,length(X1).^2);reshape(X2,1,length(X2).^2)];
    D = sum((X-repmat([x0;y0],1,length(X))).^2);
    index0=find(D>r1.^2+1.414);
    index1 = find(D(index0)<r2.^2);
    index2 = find(D<r1.^2);

    plot(X(1,index0(index1)),X(2,index0(index1)),'b*','markersize',2);
    hold on;
    plot(X(1,index2),X(2,index2),'k+','markersize',2);

    data =X(:,[index0(index1),index2]);
    label=[ones(1,length(index0(index1))),-ones(1,length(index2))];

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%kernel function of two vector xs,xt
%%especially,paramater pha in radius-basis function
%%plays a important role for  accurancy of the Kernel 
%%Machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = kernelFunc(xs,xt,type)
    switch(type)
    case "rbf"
	  pha =1.4;
	  gamma = pha.^2;
	  v=exp(-(xs-xt)'*(xs-xt)/gamma);
    case "linear"
	  q=3;
	  v = (xs'*xt+1).^q;
    endswitch
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%calculate Kernel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ KH ] = kernelMatrix(dataS,dataT,type)
    rows = length(dataS);
    columns = length(dataT);
    KH = zeros(rows,columns);
    switch(type)
    case 'rbf'
	    for i=1:rows
		    for j=1:columns
		    KH(i,j)=kernelFunc(dataS(:,i),dataT(:,j),type);
		    endfor
	    endfor
    case 'linear'
	   for i=1:rows
		    for j=1:columns
		    KH(i,j)=kernelFunc(dataS(:,i),dataT(:,j),type);
		    endfor
	    endfor
    endswitch
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train kernel svm machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ w,svmIndex,alpha ] = linearKerSvm (data,label,kerneltype)
    C=100;%slack variable
    HK = kernelMatrix(data,data,kerneltype);
    H = (label'*label).*HK;
    H = H;
    q = ones(length(data),1);
    q = -q;
    A = label;
    b = [0];
    lb = zeros(length(data),1);
    ub = ones(length(data),1)*C;
    x0 = rand(length(data),1);
    OPTIONS.MaxIter = 5000;
    EPSION = 1e-8
    [x,obj,info,lambda] = qp(x0,H,q,A,b,lb,ub,OPTIONS); % quardic optimization
    svmIndex = find(abs(x)>EPSION); % the support vector index
    alpha = x;
    w=zeros(size(data(:,1)));
    for i=1:length(svmIndex)
    jj =svmIndex(i);
    w=w+x(jj,1)*label(1,jj)*data(:,jj);
    endfor

    wo=0;
    for i=1:length(svmIndex)
    jj = svmIndex(i);
    wo+=label(1,jj)-w'*data(:,jj);
    endfor
    wo=wo./length(svmIndex);
    w =[wo;w];
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%plot the max marginal separable hyperplane(int 2-D plane
%%,it is a straight  line or curve,BUT NOW not success,TODO HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotHyperPlane(data,label,svmIndex,alpha,kerneltype)
    [X1,X2] = meshgrid(0:0.2:6,0:0.2:6);
    X = [reshape(X1,1,length(X1).^2);reshape(X2,1,length(X2).^2)];
    N = length(X);
    G = zeros(N,1);

    for i=1:length(X)
    k=0;
    for j=1:length(svmIndex)
     k=k+alpha(svmIndex(j)).*label(svmIndex(j)).* kernelFunc(data(:,svmIndex(j)),X(:,i),kerneltype);	     
    endfor
    G(i,1)=k;
    endfor
    
    hold on;
    index2 = find(abs(G)<1+1e-8);    
    plot(X(1,index2),X(2,index2),'bo'); %separable line

endfunction

clear;
clc;
hold off;
%kerneltype = 'linear';
kerneltype ='rbf';
[ data,label ] =  generateData();
[ w,svmIndex,alpha ] = linearKerSvm (data,label,kerneltype);
plot(data(1,svmIndex),data(2,svmIndex),'ro','markersize',5); %plot support vector 
hold on;
legend("Positive data","Negative data","Support vetors");
title("Kernel-based Support Vector Machine for Classfication");
%plotHyperPlane(data,label,svmIndex,alpha,kerneltype);

