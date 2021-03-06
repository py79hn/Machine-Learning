## Copyright (C) 2015 Calus Peng
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%%@ kernel SVR(Support Vector Regression) model implementation for  regression @%
%%@ problem within 1-Demission original features space.		                 @%
%%@ this source code is only used for academic  purpose,if you want to use this@% 
%%@ in you academic project,please cite the source <calus peng ,py79hn@163.com>@%
%%@ or if you want to use it for business purpose, please first send a email to@%
%%@ the author for requesting  permission,or you'll be responsible for your    @%
%%@ Illegal action.                                                            @%
%%                                                                                                   @%
%%Author: Calus Peng <py79hn@163.com>                                          @%
%%Created: 2015-12-05                                                          @%
%%code can be downloaded from github:https://github.com/py79hn/Machine-Learning@%
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%generate training  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ datas ] = generateData()
  a = -5;
  b = 5;
  N = 50;
  x = linspace(-pi,pi,N);
  y = x.^3;
  noise = a*rand(1,N)+(b-a); %random noise numbers of  size 1*N range in [a,b]
  
  datas =[x',y'+noise'];
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%kernel function,xs and xt are column vectors ,e.g,[feature1,feature2,..]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = kernelFunc(xs,xt,ftype)
   switch(ftype)
     	case "rbf"
             pha =1;
       	 gama = pha.^2;
             v=exp(-(xs-xt)'*(xs-xt)/gama);
     	case "poly"
             q=3;
             v = (xs'*xt+1).^q;
      case "sigmod"
             l=0.189;
             c=1.8;
             v = tanh(l*(xs'*xt)+c);
    endswitch
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%calculate Kernel matrix,dataS and dataT are data matrix with column vectors
%%as feature vectors , regression Hessian Martix HM is of size (2*rows)*(2*columns)
%%,because for each sample point ,there are two constrainted parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ KH ] = kernelMatrix(dataS,dataT,ftype)
   drows = length(dataS);
   dcolumns = length(dataT);
   H = zeros(drows,dcolumns);
  
   for i=1:drows
	for j=1:dcolumns
	    H(i,j)=kernelFunc(dataS(:,i),dataT(:,j),ftype);
	endfor
   endfor
   M = 2*drows
   N = 2*dcolumns
   HM = zeros(M,N);
   m=1;
   n=1;
   for i=1:drows
      n = 1; % when handle new row, start from the first column
      for j=1:dcolumns
         HM(m,n) = H(i,j);
         HM(m,n+1) = -H(i,j);
         HM(m+1,n) = -H(i,j);
         HM(m+1,n+1) = H(i,j);
         n = n+2;  % skip two  columns
      endfor
    m = m+2;  %skip two rows;
   endfor
 KH=HM;
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%regression function,datas is the training data of type [x1,y1;x2,y2;
%% x3,y3;...],C:const value ;e: yibuxing error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ svmIndex,alpha,b ] = kernelSvm_regression (datas,C,err,ftype)
  H =  kernelMatrix(datas(:,1)',datas(:,1)',ftype);
  
  N = length(H);
  l = N/2;
  ee = err*ones(1,N);
  yy = zeros(1,N);
  A  = ones(1,N);
  b =[0];
  col = columns(datas);
  y = datas(:,col); % the last column is the function value y
  k =1;
  for i=1:length(y)
     yy(k) = y(i);
     yy(k+1)=-y(i);
     A(k+1) = -A(k+1);
     k+=2;
  endfor

  q = ee -yy;
  lb = zeros(1,N);
  ub = ones(1,N)*C/l;
  x0 = rand(1,N);
  OPTIONS.MaxIter = 5000;
  EPSION = 1e-8;
  [x,obj,info,lambda] = qp(x0',H,q',A,b,lb',ub',OPTIONS);
  svmIndex = find(abs(x)>EPSION); % the support vector index
  alpha = x;
  %%the flowing code aim at calculate b%%
  jj = svmIndex(floor(1+rand()*(length(svmIndex)-1))); %randomly select a number in svmIndex
  jindex = ceil(jj(+1)/2)
  K = zeros(1,N);
  s =1;
  for i = 1:N/2
     K(s)= kernelFunc(datas(:,1)(i),datas(:,1)(jindex),ftype);
     K(s+1) = -K(s);
     s=s+2;
  endfor

 b = datas(:,col)(jindex)-alpha'*K'+err;
% alpha = x(svmIndex); % only return the aplha that corresponding to support vectors
endfunction


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate function :func(x) based on the support vectors
% alpha: paramaters figured by using QP method for SVR problem
% svs:support vectors
% x: a sample point
% type: type of kernel function,'rbf'-radial basis function,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ val ] = func(alpha,datas,b,x,ftype)
 N = length(alpha);
 K = zeros(1,N);%length(alpha)==two times of the number of training datas
 s = 1;
 for i = 1:N/2
   K(s) = kernelFunc(datas(:,1)(i),x,ftype);
   K(s+1) = -K(s);
   s=s+2;
 endfor
 val = alpha'*K'+b;
endfunction



%%%%%%%%%%%%%%%
clear;
C =2000;   %level of penalty that is set to the sample whose error is exceeding the error, the larger,the more penalty
err = 0.0001; %the larger the e value the less the support verctors
ftype="poly";%sigmod"]; % three kernel functions : poly,rbf,sigmod

hold off;
datas = generateData();
%###########using Polynomial kernel function####################%
ftype1="poly";
subplot(3,1,1);
plot(datas(:,1),datas(:,2),"k:","markersize",5);
hold on;

[svmIndex,alpha,b] = kernelSvm_regression (datas,C,err,ftype1);
plot(datas(floor((svmIndex+1)/2),1),datas(floor((svmIndex+1)/2),2),"rs","markersize",2);
hold on;

Xs= datas(:,1);
Ys = zeros(1,length(Xs));
for i=1:length(Xs)
  Ys(i) = func(alpha,datas(:,1),b,Xs(i),ftype1);
endfor
plot(Xs,Ys,"g:","markersize",5);

title("Polynomial Kernel function SVR");
legend("training data","suport vectors","predicte value");

%############using Sigmod kernel function########################%
subplot(3,1,2);
ftype2="sigmod"
plot(datas(:,1),datas(:,2),"k:","markersize",5);
hold on;

[svmIndex,alpha,b] = kernelSvm_regression (datas,C,err,ftype2);
plot(datas(floor((svmIndex+1)/2),1),datas(floor((svmIndex+1)/2),2),"rs","markersize",2);
hold on;

Xs= datas(:,1);
Ys = zeros(1,length(Xs));
for i=1:length(Xs)
   Ys(i) = func(alpha,datas(:,1),b,Xs(i),ftype2);
endfor
plot(Xs,Ys,"g:","markersize",5);

title("Sigmod kernel function SVR");
legend("training data","suport vectors","predicte value");


%##########using Radia-base kernel function######################%
subplot(3,1,3);
ftype3="rbf";
plot(datas(:,1),datas(:,2),"k:","markersize",5);
hold on;

[svmIndex,alpha,b] = kernelSvm_regression (datas,C,err,ftype3);
plot(datas(floor((svmIndex+1)/2),1),datas(floor((svmIndex+1)/2),2),"rs","markersize",2);
hold on;

Xs= datas(:,1);
Ys = zeros(1,length(Xs));
for i=1:length(Xs)
  Ys(i) = func(alpha,datas(:,1),b,Xs(i),ftype3);
endfor
plot(Xs,Ys,"g:","markersize",5);

title("Radial-base Kernel function SVR");
legend("training data","suport vectors","predicte value");





