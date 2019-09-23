% Example of how to use the mklreg
function housing_data_example()

close all
clc

addpath('simplemkl')
addpath('SVM-KM')
addpath('..');

rng(10)

%-------------------creating and plotting data----------------
DATA = load('housing_data.mat');
n=length(DATA.y);

folds=10;
y_test_all=[];
y_pred_all=[];
y_pred_null_all=[];
kernel_weights_all=[];

CVO = cvpartition(n,'k',folds);
for current_fold=1:folds
    fprintf('\nStarting fold %i of %i\n',current_fold,folds);
    
    TRAIN_ind  = find(CVO.training(current_fold));
    TEST_ind = find(CVO.test(current_fold));
    
    X_raw = DATA.X;
    y_train = DATA.y(TRAIN_ind);
    y_test = DATA.y(TEST_ind);
    y_test_all=[y_test_all;y_test];

    %----------------------Learning Parameters -------------------
    
    verbose=0;
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
    
    % parameters
    params= [];
    params.kernels = {'correlation_spearman','linear','polynomial_2','polynomial_3','polynomial_4','radial_0.001','radial_0.01','radial_0.1','radial_0.5','radial_1','radial_5','radial_10','radial_100','radial_500','correlation_pearson'};
    params.preprocessor = 'StandardScaler';%None';%'StandardScaler';
    params.kernel_normalization='CenterAndNormalize';%'CenterAndNormalize';
    
    % transformer
    X = feature_transformer(params.preprocessor,X_raw,TEST_ind);
    % kernels
    K = zeros(n,n,length(params.kernels));
    for i=1:length(params.kernels)
        K(:,:,i) = compute_kernel(X,params.kernels{i});
    end
    % scaler
    [K_train,K_test] = center_and_normalize_kernel(params,K,TRAIN_ind,TEST_ind);
    
    C_vec=[0.01,0.1,1,10,100,500];
    err=[];
    lab=[];
    betas=[];
    ypreds=[];
    for C = C_vec
        % train model
        [beta,alpha_sv,b,pos,history,obj,status] = mklsvm(K_train,y_train,C,options,verbose);        
        betas=[betas;beta];        
        alpha = zeros(length(y_train),1);
        alpha(pos) = alpha_sv;        
        ktest_final = zeros(length(TEST_ind),length(TRAIN_ind));
        ktrain_optimal = zeros(length(TRAIN_ind),length(TRAIN_ind));
        for i = 1:size(K,3)
            ktest_final = ktest_final + beta(i)*K_test(:,:,i);
            ktrain_optimal = ktrain_optimal + beta(i)*K_train(:,:,i);
        end        
        ypred = ((ktest_final*alpha)+b); % add mean from the training set
        yapp_pred = ((ktrain_optimal*alpha)+b); % add mean from the training set
        ypreds=[ypreds,ypred];
        err(end+1)=var(ypred-y_test);
        ypred_null = ones(size(ypred))*mean(y_train);        
        fprintf('FOLD %i: C = %f, error %f, R2 %f, error ratio %f\n',current_fold,C,err(end),corr2(y_test,ypred),var(y_test-ypred)/var(y_test-ypred_null));        
    end        
    [~,best_k]=min(err);        
    kernel_weights_all = [kernel_weights_all;betas(best_k,:)];    
    y_pred_all=[y_pred_all;ypreds(:,best_k)];
    y_pred_null_all = [y_pred_null_all;ypred_null];
end

results = [];
results.error = sqrt(mean((y_pred_all-y_test_all).^2));
results.error_null = sqrt(mean((y_pred_null_all-y_test_all).^2));
results.error_ratio = results.error/results.error_null; % RMSE ratio
results.error_ratio_mse = (results.error_ratio)^2; % MSE ratio (variance reduction)
results.R = corr2(y_pred_all,y_test_all);
results.R2 = (corr2(y_pred_all,y_test_all))^2;

fprintf('\n---- Final results over %i folds: Error %f (dummy %f), R %f, R2 %f, Error ratio %f ----\n',folds,results.error,results.error_null,results.R,results.R2,results.error_ratio);

figure;
subplot(2,1,1);
bar(mean(kernel_weights_all)');
axis tight;
set(gca,'xtick',1:length(params.kernels),'xticklabels',params.kernels,'ticklabelinterpreter','none','xticklabelrotation',90);
ylabel('weight');

subplot(2,1,2);
plot(1:n,y_test_all,1:n,y_pred_all);
axis tight;
%legend(lab,'location','best');

end

function kernel = compute_kernel(X,kerneltype)
% compute requested kernel, default scaling parameters according to sklearn
% for all kernels, larger values equals to higher sample similarity!
X = double(X); % convert to double, no need to save space for kernels here!
L = size(X,2); % how many features
ind = strfind(kerneltype,'_');
if ~isempty(ind)
    % kernel contains a parameter, extract it
    param_str = kerneltype(ind+1:end);
    kerneltype=kerneltype(1:(ind-1));
end
if strcmp(kerneltype,'linear') %  k(x,y) = x*y'
    kernel = X*X';
elseif strcmp(kerneltype,'radial') % k(x,y) = exp(-param*(norm(x-y)^2)/L)
    XXh1 = sum(X.^2,2)*ones(1,size(X,1));
    XXh2 = sum(X.^2,2)*ones(1,size(X,1));
    omega = XXh1+XXh2' - 2*(X*X');
    kernel = exp(-omega*str2double(param_str)/L); %
elseif strcmp(kerneltype,'polynomial') %  k(x,y) = (1+x*y'/L)^param
    kernel = (1.0 + (X*X')/L).^str2double(param_str);
elseif strcmp(kerneltype,'correlation') %  k(x,y) = corr(x,y,'type',param_str)
    kernel = corr(X',X','type',param_str);
else
    error('Unknown kernel! Allowed ones: ''linear'', ''radial_X'', ''correlation'', ''polynomial_X''');
end
end

function data = feature_transformer(preprocessor,data,test_samples)
% Feature transformation, learn from training data and apply to test data
train_samples = 1:size(data,1);
train_samples(test_samples)=[];
% make sure ok, 
assert(length(train_samples)+length(test_samples)==size(data,1));

% transform features
if strcmp(preprocessor,'StandardScaler')
    [data(train_samples,:),mu,sigma] = zscore(data(train_samples,:)); % learn params, applyt to training data
    data(test_samples,:) = (data(test_samples,:)-mu)./sigma; % apply to testing data
elseif strcmp(preprocessor,'MaxAbsScaler')
    mu = mean(data(train_samples,:));
    data = data - mu;
    sigma = max(abs(data(train_samples,:)));
    data(test_samples,:) = data(test_samples,:)./sigma;
elseif strcmp(preprocessor,'None')    
    % no preprocessing, use data as is (not generally recommended)
elseif strcmp(preprocessor,'Dummy')    
    % no preprocessing, FOR DEBUGGING PURPOSES (modify indexing, but not data)
else        
    error('Unknown preprocessing type (%s), only ''None'', ''StandardScaler'' and ''MaxAbsScaler'' allowed!',preprocessor);
end

end

function [X_train,X_test] = center_and_normalize_kernel(params,X,tr_idx,te_idx)
% kernel normalization and splitting into test and train sets
% References:
%   vis.caltech.edu/~graf/my_papers/proceedings/GraBor01.pdf
%   Kernel Methods for Pattern Analysis, Shawe-Taylor & Cristianini, Campbridge University Press, 2004

% Centering and normalizing (Pronto-style, recommended!)
if strcmp(params.kernel_normalization,'CenterAndNormalize')
    % See Pronto function "prt_apply_operations.m"
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
    X_tt = X(te_idx, te_idx,:);
    tr = 1:length(tr_idx);
    te = (1:length(te_idx))+max(tr);
    for d = 1:size(X,3)
        % Centering: to make the dataset zero mean; kernel will capture the co-variance of data items around the mean        
        [X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d)] = prt_centre_kernel(X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d));
        Phi = [X_train(:,:,d), X_test(:,:,d)'; X_test(:,:,d), X_tt(:,:,d)];
        % Normalization: projects the data on the unit sphere, removes the effect of the length of data vectors
        Phi = prt_normalise_kernel(Phi);
        X_train(:,:,d) = Phi(tr,tr);
        X_test(:,:,d) = Phi(te,tr);
    end
% only centering
elseif strcmp(params.kernel_normalization,'Center')
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
    for d = 1:size(X,3)
        [X_train(:,:,d),X_test(:,:,d)] = prt_centre_kernel(X_train(:,:,d),X_test(:,:,d));
    end
% simple centering and trace normalization
elseif strcmp(params.kernel_normalization,'CenterAndTrace')
    n = size(X,1);
    G = (eye(n) - ones(n)/n);
    for d = 1:size(X,3)
        X(:,:,d) = G*X(:,:,d)*G; % centering, not respecting train/test split!
        % divide by trace
        X(:,:,d) = n*X(:,:,d)/trace(X(:,:,d));
    end
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
% do nothing, just split
elseif strcmp(params.kernel_normalization,'None')
    % no scaling, just splitting (NOT recommended!)
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
else
    error('Unknown kernel normalization type (%s) !!!!',params.kernel_normalization);
end

end