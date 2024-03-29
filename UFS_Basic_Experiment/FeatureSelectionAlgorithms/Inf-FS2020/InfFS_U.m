function [RANKED, WEIGHT, SUBSET] = InfFS_U( X_train, alpha, verbose )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the Auto-UFSTool, which is an Automatic Unspervised %
% Feature Selection Toolbox                                                %
% Version 1.0 August 2022   |  Copyright (c) 2022   | All rights reserved  %
%        ------------------------------------------------------            %
% Redistribution and use in source and binary forms, with or without       %
% modification, are permitted provided that the following conditions are   %
% met:                                                                     %
%    * Redistributions of source code must retain the above copyright      %
%    * Redistributions in binary form must reproduce the above copyright.  %
%         ------------------------------------------------------           %
% "Auto-UFSTool,                                                           %
%  An Automatic MATLAB Toolbox for Unsupervised Feature Selection"         %
%         ------------------------------------------------------           %
%                    Farhad Abedinzadeh | Yegane Modaresnia                %
%          farhaad.abedinzade@gmail.com | y.modaresnia@yahoo.com           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% [RANKED, WEIGHT, SUBSET ] = InfFS_U( X_train, alpha, verbose ) computes ranks and weights
% of features for input data matrix X_train using Inf-FS algorithm.
%
% Version 5.1, June 2020.
%
% INPUT:
%
% X_train is a T by n matrix, where T is the number of samples and n the number
% of features.样本*特征
% alpha: mixing coefficient in range [0,1]
% Verbose, boolean variable [0, 1]
%
% OUTPUT:
%
% RANKED are indices of columns in X_train ordered by attribute importance,
% meaning RANKED(1) is the index of the most important/relevant feature.
% WEIGHT are attribute weights with large positive weights assigned
% to important attributes.
% SUBSET is the selected subset of features 选定的特征子集
%  ------------------------------------------------------------------------
% Please consider to cite:

%[1] Roffo, G., Melzi, S., Castellani, U., Vinciarelli, A., and Cristani, M., 2020. Infinite Feature Selection: A Graph-based Feature Filtering Approach. In the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

%[1] Roffo, G., Melzi, S., Castellani, U. and Vinciarelli, A., 2017. Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach. arXiv preprint arXiv:1707.07538.

%[2] Roffo, G., Melzi, S. and Cristani, M., 2015. Infinite feature selection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 4202-4210).

%  ------------------------------------------------------------------------

%% The Inf-FS method
% fprintf('\n+ Feature selection method: Inf-FS Unsupervised (2020)\n');

if (nargin<3)
    verbose = 0;
end

%% 1) Standard Deviation over the samples
if (verbose)
    fprintf('1) Priors/weights estimation \n');
end

[ corr_ij, pval ] = corr( X_train, 'type','Spearman' );
corr_ij(isnan(corr_ij)) = 0; % remove NaN
corr_ij(isinf(corr_ij)) = 0; % remove inf
corr_ij =  1-abs(corr_ij);

% Standard Deviation Est.
STD = std(X_train,[],1);
STDMatrix = bsxfun( @max, STD, STD' );
STDMatrix = STDMatrix - min(min( STDMatrix ));
sigma_ij = STDMatrix./max(max( STDMatrix ));
sigma_ij(isnan(sigma_ij)) = 0; % remove NaN
sigma_ij(isinf(sigma_ij)) = 0; % remove inf

%% 2) Building the graph G = <V,E>
if (verbose)
    fprintf('2) Building the graph G = <V,E> \n');
end

N = size(X_train,2); %N = 65
eps = 5e-06 * N;
factor = 1 - eps; % shrinking 

A =  ( alpha*sigma_ij + (1-alpha)*corr_ij );

rho = max(sum(A,2));

% Substochastic Rescaling "Substochastic Rescaling" 可以翻译为 "次随机重标定" 或 "次随机缩放"。
% 这是一个数学术语，通常表示将矩阵或概率分布的元素进行调整，以确保它们的总和小于等于1，从而创建一个次随机矩阵或概率分布。
A = A ./ ( max(sum(A,2))+eps);

assert(max(sum(A,2)) < 1.0);

%% Letting paths tend to infinite: Inf-FS Core
I = eye( size( A ,1 )); % Identity Matrix


r = factor/rho;  

y = I - ( r * A );

S = inv( y );


%% 5) Estimating energy scores
WEIGHT = sum( S , 2 ); % prob. scores s(i)

%% 6) Ranking features according to s
[~ , RANKED ]= sort( WEIGHT , 'descend' );

RANKED = RANKED';
WEIGHT = WEIGHT';

e = ones(N,1);
t = S * e;

nbins = 0.5*N;
counts = hist(t,nbins);

thr = mean(counts);
size_sub = sum(counts>thr);


fprintf('Inf-FS (U) Nb. Features Selected = %.4f (%.2f%%)\n',size_sub,100*(size_sub/N))

SUBSET = RANKED(1:size_sub);


end

%  =========================================================================
%   More details:
%   Reference   : Infinite Feature Selection: A Graph-based Feature Filtering Approach
%   Journal     : IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
%   Author      : Roffo, G., Melzi, S., Castellani, U., Vinciarelli, A., and Cristani, M.
%   Link        : https://github.com/giorgioroffo/Infinite-Feature-Selection
%  =========================================================================
