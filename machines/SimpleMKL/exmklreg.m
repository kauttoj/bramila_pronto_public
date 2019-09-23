% Example of how to use the mklreg
%

clear all
close all
clc

addpath('simplemkl')
addpath('SVM-KM')

rng(10)

%-------------------creating and plotting data----------------
n=150;
bruit=0.3;
freq=0.8;
x=linspace(0,8,n)';
y=cos(exp(freq*x)) +randn(n,1)*bruit;

TEST_ind = sort(randsample(n,20));
TRAIN_ind = 1:n;
TRAIN_ind(TEST_ind)=[];

xapp = x(TRAIN_ind,:);
yapp = y(TRAIN_ind);

xtest = x(TEST_ind,:);
ytest=y(TEST_ind);

%----------------------Learning Parameters -------------------
C = 150; 
verbose=1;
options.algo='svmreg';
options.seuildiffsigma=1e-3;
options.seuildiffconstraint=0.01;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-2;
options.numericalprecision=1e-8;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=300;
options.seuil=0;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.verbosesvm=0;
options.svmreg_epsilon=0.01;
options.efficientkernel=0;
optionK.pow=0;

kernelt={'linear','gaussian' 'poly'};
kerneloptionvect={[],[0.005:0.04:0.2 0.5 1 2 5 7 10 12 15 17 20 22],[1 2 3 4]};
variablevec={'all' 'all' 'all'};
dim=size(xapp,2);

[kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
[K]=mklbuildkernel(x,kernel,kerneloptionvec,[],[],optionK);
[K,optionK.weightK]=WeightK(K);

K_train = K(TRAIN_ind,TRAIN_ind,:);
K_test = K(TEST_ind,TRAIN_ind,:);

[beta,alpha_sv,b,pos,history,obj,status] = mklsvm(K_train,yapp,C,options,verbose);

alpha = zeros(length(yapp),1);
alpha(pos) = alpha_sv;

ktest_final = zeros(length(TEST_ind),length(TRAIN_ind));
ktrain_optimal = zeros(length(TRAIN_ind),length(TRAIN_ind));
for i = 1:size(K,3)
    ktest_final = ktest_final + beta(i)*K_test(:,:,i);
    ktrain_optimal = ktrain_optimal + beta(i)*K_train(:,:,i);
end

ypred = ((ktest_final*alpha)+b); % add mean from the training set
yapp_pred = ((ktrain_optimal*alpha)+b); % add mean from the training set
ypred_null = ones(size(ypred))*mean(yapp);

plot(xtest,ytest,'b',xapp,yapp,'r-+',xapp,yapp_pred,'c-+',xtest,ypred,'g');
legend('test','train','train_pred','pred')

fprintf('error %f, R2 %f, error ratio %f\n',norm(ytest-ypred),corr2(ytest,ypred),norm(ytest-ypred)/norm(ytest-ypred_null));

% [kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
% [K]=mklbuildkernel(xapp,kernel,kerneloptionvec,[],[],optionK);
% [K,optionK.weightK]=WeightK(K);   
% 
% [beta,w,b,posw,story,obj] = mklsvm(K,yapp,C,options,verbose);
% kerneloption.matrix=mklbuildkernel(xtest,kernel,kerneloptionvec,xapp(posw,:),beta,optionK);
% 
% ypred=svmval([],[],w,b,'numerical',kerneloption);
% plot(xtest,ytest,'b',xapp,yapp,'r',xapp,yapp,'r+',xtest,ypred,'g')
% 
% fprintf('error %f\n',norm(ytest-ypred))