% Write a cell matrix to a text file

function fid = text_file_write(file_name, cell_mat, ind_date)
% Inputs:
%   file_name: a string of the write file name (e.g. 'data.txt')
%   cell_mat: a cell matrix of data (first row is title; subsequent rows are main body)
%   ind_date: a 0/1 indicator if the first column is snap date
% Outputs:
%   fid: an indicator (not used)

[nrows,ncols]= size(cell_mat);

%% Specify the format string
fm_title = []; %format for first row of title
fm_main_body = [];  %format for main body
for j = 1:ncols
    fm_title = [fm_title, '%s '];
    
    if ind_date == 0 %first column is data        
        fm_main_body = [fm_main_body, '%d '];
    else %first column is snap date
        if j == 1
            fm_main_body = [fm_main_body, '%s '];
        else
            fm_main_body = [fm_main_body, '%d '];
        end
    end
end
fm_title = [fm_title, '\n'];
fm_main_body = [fm_main_body, '\n'];

%% Write the file
fid = fopen(file_name, 'w');
fprintf(fid, fm_title, cell_mat{1,:});
for row=2:nrows
%     fprintf(fid, '%s %d %d %d\n', mycell{row,:});
    fprintf(fid, fm_main_body, cell_mat{row,:});
end
fclose(fid);
