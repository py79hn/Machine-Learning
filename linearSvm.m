## Copyright (C) 2015 Calus Peng
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%%@ linear svm model implementation demo for  classfication problem            @%
%%@ this source code is only used for academic  purpose,if you want to use this@% 
%%@ in you academic project,please cite the source <calus peng ,py79hn@163.com>@%
%%@ or if you want use it for business purpose, please first send a email to   @%
%%@ the author for requesting  permission,or you'll be responsible for your    @%
%%@ Illegal action.                                                            @%
%%Author: Calus Peng <py79hn@163.com>                                          @%
%%Created: 2015-12-02                                                          @%
%%code can be downloaded from github:https://github.com/py79hn/Machine-Learning@%
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%genarate the training data ,positie sample are assgin label +1,while the
%%negative samples are assigned the label -1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ data,label ] =generatData()
    N=60; %number of training data
    x1 = rand(1,N)*(3-1)+1;
    y1 = rand(1,N)*(5-3)+3;
    x2 = rand(1,N)*(5-3)+3;
    y2 = rand(1,N)*(3-1)+1;

    x =[x1,x2];
    %x =(x- mean(x))./var(x);
    y =[y1,y2];
    %y =(y- mean(y))./var(x);
    data =[x;y];
    label = [ones(size(x1)),-ones(size(x2))];
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot the  traing data belonging  to 2-classes using different colors,and%
%the support vector with black circle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotData( data,label,svmIndex)
index1 = find(label ==1);
index2 = find(label ==-1);
hold off;
plot(data(1,index1),data(2,index1),'g+');
hold on;
plot(data(1,index2),data(2,index2),'b*');
plot(data(1,svmIndex),data(2,svmIndex),'rs');
hold off;
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%traing svm model using QP algorithm to calculate the weight w and wo%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ w,wo,svmIndex ] = linearSvm (data,label)
    C=NaN;%slack variable
    H = (label'*label).*(data'*data);
    H = H;
    q = ones(length(data),1);
    q = -q;
    A = label;
    b = [0];
    lb = zeros(length(data),1);
    ub = ones(length(data),1)*C;
    x0 = rand(length(data),1);
    OPTIONS.MaxIter = 1000;
    EPSION = 1e-8;
    [x,obj,info,lambda] = qp(x0,H,q,A,b,lb,ub,OPTIONS); % quardic optimization      
    svmIndex = find(abs(x)>EPSION); % the support vector index

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
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%draw straight line using w(1)*x1+w(2)*x2+wo={-1,+1,0} ,namyly 
%% w'*x+wo={-1,+1,0}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function drawLine(w,wo)
    title('SVM for two-classes Classfication ');
    xlabel('x1');
    ylabel('x2');
    legend("Positive","Negative");
    x1=linspace(0,5,50);
    %x1 = (x1-mean(x1))./var(x1);

    ye0=(0-w(1)*x1-wo)/w(2);
    %ye0 = (ye0-mean(ye0))./var(ye0);

    yem1=(-1-w(1)*x1-wo)/w(2);
    %yem1 = (yem1-mean(yem1))./var(yem1);

    yep1=(1-w(1)*x1-wo)/w(2);
    %yep1 = (yep1-mean(yep1))./var(yep1);

    plot(x1,ye0,'k-',x1,yem1,'c:',x1,yep1,'c:');
endfunction

clear;
clc;

[ data,label ] =generatData();
[ w,wo,svmIndex ] = linearSvm(data,label);
plotData(data,label,svmIndex);

hold on;
drawLine(w,wo);


