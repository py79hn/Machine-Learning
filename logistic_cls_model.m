## Copyright (C) 2015 Calus Peng
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%
%%@ logistic  model implementation demo for muti-class classfication problem   @%
%%@ this source code is only used for academic  purpose,if you want to use this@% 
%%@ in you academic project,please site the source <calus peng ,py79hn@163.com>@%
%%@ or if you want to use it for business purpose, please first send a email to@%
%%@ the author for requesting  permission,or you'll be responsiable for your   @%
%%@ illegal  actiities.                                                        @%
%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

## Author: calus peng <py79hn@163.com>
## Created: 2015-11-04

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%function to genarate the traing data,
%%K- how many classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
function [label,data] = generateData(K)
    many =20;
    a=zeros(K,many);
    b=zeros(K,many);
    
   p1=[1.0,3.0;2.5,4;0.0,1.5];
   p2=[0.5,2.0;2.5,4;2.5,3.5];
    for i=1:K
	 a(i,:)=rand(1,many)*(p1(i,2)-p1(i,1))+p1(i,1);
         b(i,:)=rand(1,many)*(p2(i,2)-p2(i,1))+p2(i,1);
    endfor
 
    aa = [a(1,:),a(2,:),a(3,:)];
    bb = [b(1,:),b(2,:),b(3,:)];
    data = [aa;bb];
    label=zeros(length(data),K);
    start=1;
    for i=1:K
	label(start:start+length(a(i,:))-1,i)=1;
        start=start+length(a(i,:));
    endfor
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute each iterator training step cross entropy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ entropy ] = computeEnropy(w,data,label)
    resultT = w*data;
    sumexpT = sum(exp(resultT));
    predictT = exp(resultT)./repmat(sumexpT,rows(resultT),1);
   entropy = -sum(sum(log(predictT).*label'));   %calculate cross entropy
endfunction


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% paramaters finitialization for logistic classfication model%%
%%K: how many classes
%%D: feature dimension
%%LOOP: training iterators numbers
%%RATE:learning rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
K=3;
D=2;
LOOP=500;
count=1;
RATE=0.01;
[label,data] = generateData(K);
data =[ones(1,length(data));data]; %composite the training data
osum = zeros(1,K);
predict=zeros(1,K);

corssEntropys= zeros(1,LOOP); % store each iterator cross entropy
w=(rand(K,D+1)*2-1)./100;     %initlizate the weight w(K,D+1) in range(-0.01,0.01)

while (count<=LOOP)
  dw=zeros(K,D+1);
  for t=1:length(data)    % iterate on all training data
     for i=1:K
	  osum(i)=0;
	for j=1:D+1
	 osum(i)=osum(i)+w(i,j)*data(j,t);
	endfor
     endfor
       predict=exp(osum)./sum(exp(osum));
     for j=1:D+1
	dw(:,j)=dw(:,j)+(label(t,:)'-predict').*data(j,t);
     endfor
     w=w+RATE*dw;
  endfor
  corssEntropys(1,count) = computeEnropy(w,data,label);
  count+=1;
endwhile

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%to plot the tarined model on the 3-D coordinate system%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x=linspace(0,5,20);
y=linspace(0,5,20);
[xx,yy]=meshgrid(x,y);

tmp_x=reshape(xx,1,length(xx).^2);
tmp_y=reshape(yy,1,length(yy).^2);
tmp_1=ones(size(tmp_x));
plot_data=[tmp_1;tmp_x;tmp_y];

result=w*plot_data;       %prediction value 

sumexp=sum(exp(result));  % used for softmax

figure(1);
subplot(2,1,1);
for k=1:K
    tmp_zz=reshape(result(k,:),length(xx),length(yy));     
    surfc(xx,yy,tmp_zz./10);
    colorbar;
    hold on;
endfor

subplot(2,1,2);
plot(1:length(corssEntropys),corssEntropys);
colorbar;

figure(2); 
for k=1:K
    tmp_zz=reshape(result(k,:),length(xx),length(yy));
    surfc(xx,yy,exp(tmp_zz)./reshape(sumexp,length(xx),length(yy)));
    colorbar;
    hold on;
endfor

