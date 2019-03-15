% Read the raw data y;
% Assemble the AR lags of each target as its features

function [y, z, nof_z_vec, snap_dt] = read_process_data(read_info, vec_ARlags, m) 
% Inputs:
%   read_info: a structure with 3 fields:
%      read_info.read_file: a string of the read file name;
%      read_info.read_sheet: a string of the read sheet name;
%      read_info.read_cell: a string of the read range (excluding title in the first row; including the first column of date if it exists)
%   vec_ARlags: a m-by-1 cell of the AR lags for each target (e.g cell j is [1 2 5] for AR lags y_{j,t-1}, y_{j,t-2}, y_{j,t-5})
%   m: a scalar of the number of targets
% Outputs:
%   y: a n-by-m matrix of target data by column
%   z: a m-by-1 cell of feature data with cell j being a n-by-k_j matrix of feature data for target j
%   nof_z_vec: a m-by-1 vector of the number of features for each target
%   snap_dt: a n-by-1 cell of snap date (if the first column is not snap date, snap_dt = [])


%% Read raw data
[y_raw, snap_dt] = xlsread(read_info.read_file, read_info.read_sheet, read_info.read_cell);
[nobs_raw,m_raw] = size(y_raw);
if m_raw ~= m
    error('Error in the number of targets imported!');
end


%% Process raw data to determine targets and features (AR lags)
% Targets could have different AR lags, but data of all targets start at 1+max(AR lag) and have the same length
max_ARlag_vec = zeros(m,1);
for j = 1:m
    if isempty(vec_ARlags{j})
        max_ARlag_vec(j) = 0;
    else
        max_ARlag_vec(j) = max(vec_ARlags{j});
    end
end
max_ARlag = max(max_ARlag_vec); 
nobs = nobs_raw - max_ARlag; %all targets should have the same length

y = zeros(nobs,m); %targets
z = cell(m,1); %features, including constant
nof_z_vec = zeros(m,1);
for j = 1:m
    nof_ARlags = length(vec_ARlags{j});

    y(:,j) = y_raw(max_ARlag+1:nobs_raw,j); %target
    
    if nof_ARlags > 0
        z_nonconst = zeros(nobs, nof_ARlags);
        for i = 1:nof_ARlags
            z_nonconst(:,i) = y_raw(max_ARlag+1-vec_ARlags{j}(i):nobs_raw-vec_ARlags{j}(i),j);
        end %non-constant features
    else
        z_nonconst = [];
    end
    z{j} = [ones(nobs,1)  z_nonconst]; %features
    nof_z_vec(j) = size(z{j},2);
end


