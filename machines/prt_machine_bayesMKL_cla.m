function output = prt_machine_bayesMKL_cla(d,def,args,args_optional)
% Run L1-norm MKL - wrapper for simpleMKL
% FORMAT output = prt_machine_sMKL_reg(d,args)
% Inputs:
%   d         - structure with data information, with mandatory fields:
%     .train      - training data (cell array of matrices of row vectors,
%                   each [Ntr x D]). each matrix contains one representation
%                   of the data. This is useful for approaches such as
%                   multiple kernel learning.
%     .test       - testing data  (cell array of matrices row vectors, each
%                   [Nte x D])
%     .tr_targets - training labels (for classification) or values (for
%                   regression) (column vector, [Ntr x 1])
%     .use_kernel - flag, is data in form of kernel matrices (true) of in 
%                form of features (false)
%    args     - simpleMKL arguments
% Output:
%    output  - output of machine (struct).
%     * Mandatory fields:
%      .predictions - predictions of classification or regression [Nte x D]
%     * Optional fields:
%      .func_val - value of the decision function
%      .type     - which type of machine this is (here, 'classifier')
%      .
%__________________________________________________________________________
% Copyright (C) 2011 Machine Learning & Neuroimaging Laboratory

% Written by J. Mourao-Miranda 

tr_targets = d.tr_targets; % mean centre targets

%ktrain = d.train;
%ktest = d.test;
%reshape previously normalized kernel
ktrain = zeros(size(d.train,2),size(d.train,1),size(d.train,3));
ktest = zeros(size(d.train,2),size(d.test,1),size(d.train,3));
for k = 1:size(d.train,3)
    ktrain(:,:,k) =  (d.train(:,:,k))';
    ktest(:,:,k) =  (d.test(:,:,k))';
end

%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for sample weights
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for intermediate noise
parameters.alpha_upsilon = 1;
parameters.beta_upsilon = 1;

%set the hyperparameters of gamma prior used for bias
parameters.alpha_gamma = def.model.alpha_gamma;
parameters.beta_gamma = def.model.beta_gamma;

%set the hyperparameters of gamma prior used for kernel weights
parameters.alpha_omega = def.model.alpha_omega;
parameters.beta_omega = def.model.beta_omega;

%set the hyperparameters of gamma prior used for output noise
parameters.alpha_epsilon = 1;
parameters.beta_epsilon = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems (like in Nature Biotechnology paper)

%set the number of iterations
parameters.iteration = 300;

%determine whether you want to calculate and store the lower bound values
parameters.progress = 0;

%set the seed for random number generator used to initalize random variables
parameters.seed = 666;

%set the number of tasks (e.g., the number of compounds in Nature Biotechnology paper)
T = 1;
%set the number of kernels (e.g., the number of views in Nature Biotechnology paper)
P = size(ktrain,3);

state = bemkl_supervised_classification_variational_train(ktrain,tr_targets, parameters);

beta = state.be.mu((T+1):(T+P));

alpha = zeros(length(d.tr_targets),1);
%alpha(pos) = alpha_sv;

func_val = bemkl_supervised_classification_variational_test(ktest,state);

predictions = func_val;

% Outputs
%-------------------------------------------------------------------------
output.predictions = predictions;
output.func_val    = func_val;
output.type        = 'classification';
output.alpha       = alpha;
output.b           = nan;%b;
output.totalSV     = nan;%length(alpha_sv);
output.beta        = beta/sum(beta); %kernel weights

end