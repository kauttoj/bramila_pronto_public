function TEST_MKL_machines()

clc;
clearvars;
close all

rng('default')
rng(1);

addpath([pwd,filesep,'bmtmkl']);
addpath([pwd,filesep,'SimpleMKL/simplemkl']);
addpath([pwd,filesep,'SimpleMKL/SVM-KM']);

if 0
    load carsmall
    X = [Horsepower,Weight,Acceleration,Cylinders,Displacement];
    Y = MPG;
end
if 1
    Y = ([16    22    12    10    12    19    13    10    11    17    29    12    16    20    15    17    23    24    14    11    10    13    14    18    15    14    16    12    27    23    11    12    14    19    12    21    16    16    19    15    21    16    17    10    12    30    24]');
    X = ([Y,randn(size(Y,1),30*10)]);
end
bad = isnan(Y) | sum(isnan(X),2)>0;
Y(bad)=[];
X(bad,:)=[];
N = size(X,1);

Y_real = [];
Y_predicted = [];
simpleMKL_weights=[];
bayesMKL_weights=[];
for test_sample = 1:N
   
    X_train = X;
    Y_train = Y;
    
    X_train(test_sample,:)=[];
    Y_train(test_sample)=[];
    
    X_test = X(test_sample,:);
    Y_test =Y(test_sample);
    
    m = mean(Y_train);
    Y_train = Y_train - m;
    
    if 1
        [X_train,mu,sigma] = zscore(X_train);
        X_test = (X_test - mu)./sigma;
    end
    
    XX = [X_train;X_test];
    
    tr_idx = 1:size(X_train,1);
    te_idx = size(X_train,1)+1;
    
    if 0
        K=zeros(size(XX,1),size(XX,1),4);
        K(:,:,1) = build_standarized_kernel(XX,XX,'linear');
        K(:,:,2) = build_standarized_kernel(XX,XX,'polynomial',2);
        K(:,:,3) = build_standarized_kernel(XX,XX,'polynomial',3);
        K(:,:,4) = build_standarized_kernel(XX,XX,'gaussian',gamma);
    else
        nn=(size(XX,2)-1)/10;
        K=zeros(size(XX,1),size(XX,1),nn*2+2);
        k=1;
        K(:,:,k) = compute_kernel(XX(:,1),'linear');        
        k=2;
        K(:,:,k) = compute_kernel(XX(:,1),'radial_1');
        kk=2;
        for k=1:nn
            kk=kk+1;
            K(:,:,kk) = compute_kernel(XX(:,1+((k-1)*10)+(1:10)),'radial_1');
            kk=kk+1;
            K(:,:,kk) = compute_kernel(XX(:,1+((k-1)*10)+(1:10)),'linear');
        end
    end
    
    if 0
        K_train = K(1:end-1,1:end-1,:);
        K_test = K(end,1:end-1,:);
        %K_test = permute(K(end,1:end-1,:),[2,1,3]);
    else
        [K_train,K_test] = center_and_normalize_kernel('CenterAndNormalize',K,tr_idx,te_idx);
    end
%     
%     K_test=zeros(size(X_train,1),size(X_test,1),4);
%     K_test(:,:,1) = kernel_function(X_train,X_test,'linear');
%     K_test(:,:,2) = kernel_function(X_train,X_test,'polynomial',2);
%     K_test(:,:,3) = kernel_function(X_train,X_test,'polynomial',3);
%     K_test(:,:,4) = kernel_function(X_train,X_test,'gaussian',gamma);
    
    [func_val,weights] = simpleMKL(K_train,Y_train,K_test,50);
    simpleMKL_weights=[simpleMKL_weights;weights];
    Y_predicted_simpleMKL(test_sample) = func_val + m;
    
    K_test = permute(K_test,[2,1,3]);
    [func_val,weights] = bayesMKL(K_train,Y_train,K_test);         
    bayesMKL_weights=[bayesMKL_weights;weights];
    Y_predicted_bayesMKL(test_sample) = func_val + m;
    
    %OUTPUT = train_mRVM2_MKL('-p',X_train,Y_train,0,1,100,,kernel_param,plot_flag,dataset_name)    
    
    Y_real(test_sample) = Y_test;
    
end
x = 1:length(Y_real);
plot(x,Y_real,x,Y_predicted_simpleMKL,x,Y_predicted_bayesMKL)
legend('real','simpleMKL','bayesMKL');

figure;
mat = [sum(abs(simpleMKL_weights));sum(abs(bayesMKL_weights))]';
bar(mat./sum(mat));
legend('simpleMKL','bayesMKL');

fprintf('Error: %f (simpleMKL), %f (bayesMKL)\n',sum((Y_real-Y_predicted_simpleMKL).^2),sum((Y_real-Y_predicted_bayesMKL).^2));
fprintf('R2: %f (simpleMKL), %f (bayesMKL)\n',corr2(Y_real,Y_predicted_simpleMKL),corr2(Y_real,Y_predicted_bayesMKL));

end



function [X_train,X_test] = center_and_normalize_kernel(params,X,tr_idx,te_idx)
% kernel normalization and splitting into test and train sets
% References:
%   vis.caltech.edu/~graf/my_papers/proceedings/GraBor01.pdf
%   Kernel Methods for Pattern Analysis, Shawe-Taylor & Cristianini, Campbridge University Press, 2004

% Standard centering and normalizing (Pronto-style, recommended!)
if strcmp(params,'CenterAndNormalize')
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
elseif strcmp(params,'Center')
    % See Pronto function "prt_apply_operations.m"
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
    X_tt = X(te_idx, te_idx,:);
    tr = 1:length(tr_idx);
    te = (1:length(te_idx))+max(tr);
    for d = 1:size(X,3)
        % Centering: to make the dataset zero mean; kernel will capture the co-variance of data items around the mean        
        [X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d)] = prt_centre_kernel(X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d));
    end    
% alternative normalization, for testing
elseif strcmp(params,'CenterAndTrace')
    n = size(X,1);
    G = (eye(n) - ones(n)/n);
    for d = 1:size(X,3)
        X(:,:,d) = G*X(:,:,d)*G; % centering, not respecting train/test split!
        % divide by trace
        X(:,:,d) = n*X(:,:,d)/trace(X(:,:,d));
    end
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
elseif strcmp(params,'Trace')
    for d = 1:size(X,3)
        X(:,:,d) = X(:,:,d)/trace(X(:,:,d));
    end
    X = X*size(X,1);
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);    
elseif strcmp(params,'None')
    % no scaling, just splitting (NOT recommended!)
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
else
    error('Unknown kernel normalization type (%s) !!!!',params);
end

end


function kernel = compute_kernel(X,kerneltype)
% compute requested kernel, default scaling parameters according to sklearn
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

function [func_val,weights] = bayesMKL(ktrain,tr_targets,ktest)

%initalize the parameters of the algorithm
parameters = struct();
%set the hyperparameters of gamma prior used for sample weights
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;
%set the hyperparameters of gamma prior used for intermediate noise
parameters.alpha_upsilon = 1;
parameters.beta_upsilon = 1;
%set the hyperparameters of gamma prior used for bias
parameters.alpha_gamma = 1;
parameters.beta_gamma = 1;
%set the hyperparameters of gamma prior used for kernel weights
parameters.alpha_omega = 1e-6;
parameters.beta_omega = 1e+6;
%set the hyperparameters of gamma prior used for output noise
parameters.alpha_epsilon = 1;
parameters.beta_epsilon = 1;
%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems (like in Nature Biotechnology paper)
%set the number of iterations
parameters.iteration = 200;
%determine whether you want to calculate and store the lower bound values
parameters.progress = 0;
%set the seed for random number generator used to initalize random variables
parameters.seed = 666;
%set the number of tasks (e.g., the number of compounds in Nature Biotechnology paper)
state = bemkl_supervised_regression_variational_train(ktrain,tr_targets,parameters);
res = bemkl_supervised_regression_variational_test(ktest,state);
func_val = res.y.mu;
weights = abs(state.be.mu(2:end))';
end

function [func_val,weights] = simpleMKL(ktrain,tr_targets,ktest,C_opt)

prt_def=[];
prt_def.model.svmargs     = 1;
prt_def.model.libsvmargs  = '-q -s 0 -t 4 -c ';
prt_def.model.rtargs      = 601;
prt_def.model.l1MKLargs   = 1; % how many arguments
prt_def.model.l1MKLmaxitr = 250;
def = prt_def;

%------------------------------------------------------
% configure simpleMKL options
%------------------------------------------------------
verbose=0;
options.algo='svmreg'; % Choice of algorithm in mklsvm can be either
% 'svmclass' or 'svmreg'

%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
% are set to zero
options.lambdareg = 1e-8;          % ridge added to kernel matrix

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=def.model.l1MKLmaxitr;             % maximal number of iteration
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one

options.miniter=0;                 % minimal number of iterations
options.verbosesvm=0;              % verbosity of inner svm algorithm
options.efficientkernel=0;         % use efficient storage of kernels
options.svmreg_epsilon=0.001;

% Run simpleMKL
%--------------------------------------------------------------------------
options.sigmainit = ones(1,size(ktrain,3))/size(ktrain,3); %initialize kernel weights

[beta,alpha_sv,b,pos,history,obj,status] = mklsvm(ktrain,tr_targets,C_opt,options,verbose);

alpha = zeros(length(tr_targets),1);
alpha(pos) = alpha_sv;

ktest_final = zeros(length(tr_targets),size(ktest,1));
for i = 1:size(ktest,3)
    ktest_final = ktest_final + beta(i)*ktest(:,:,i)';
end

func_val = ((ktest_final'*alpha)+b); % add mean from the training set
weights=beta;

end
