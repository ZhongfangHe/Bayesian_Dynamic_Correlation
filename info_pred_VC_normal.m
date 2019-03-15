


function write_file_pred = info_pred_VC_normal(logp_vec, pred_para_est, pred_mh, snap_dt, write_path) %store est info to check if anything goes wrong

[nof_pred, k] = size(pred_mh);
m = k/2 - 1;

name_absum = cell(m,1);
name_ap = cell(m,1);
name_ar_absum = cell(m,1);
name_ar_ap = cell(m,1);
for j = 1:m
    name_absum{j} = ['absum_', num2str(j)];
    name_ap{j} = ['ap_', num2str(j)];
    name_ar_absum{j} = ['AccptRate_absum_', num2str(j)];
    name_ar_ap{j} = ['AccptRate_ap_', num2str(j)];    
end
para_name = [{'log_pred_density'}; name_absum; name_ap; {'efsum'}; {'ep'}; name_ar_absum; name_ar_ap; {'AccptRate_efsum'}; {'AccptRate_ep'}];

write_file_pred = [write_path, 'Forecast.csv'];
if isempty(snap_dt) == 0 %first column is snap date
    ind_date = 1;
    write_mat = cell(nof_pred+1, 6+4*m);
    write_mat{1,1} = 'Date';
    write_mat(1,2:(6+4*m)) = para_name';
    nobs_raw = length(snap_dt);
    write_mat(2:(nof_pred+1),1) = snap_dt((nobs_raw-nof_pred+1):nobs_raw);
    write_mat(2:(nof_pred+1),2:(6+4*m)) = num2cell([logp_vec  pred_para_est  pred_mh]);     
else %first column is numbers, not snap dates
    ind_date = 0;
    write_mat = cell(nof_pred+1, 5+4*m);
    write_mat(1,:) = para_name';
    write_mat(2:(nof_pred+1),:) = num2cell([logp_vec  pred_para_est  pred_mh]);    
end
text_file_write(write_file_pred, write_mat, ind_date);
