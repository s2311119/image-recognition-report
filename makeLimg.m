function makeLimg()
    %pos
    list = textread('pos_urllist.txt', '%s');
    OUTDIR1 = 'posImgDir_easy';
    OUTDIR2 = 'posImgDir_diff';
    mkdir(OUTDIR1);
    mkdir(OUTDIR2);
    for i = 1:size(list, 1)
        fname1 = strcat(OUTDIR1,'/',num2str(i,'%04d'),'.jpg');
        fname2 = strcat(OUTDIR2,'/',num2str(i,'%04d'),'.jpg');
        websave(fname1,list{i});
        copyfile(fname1, fname2);
    end
    % neg(簡単な方)
    path = fullfile('/MATLAB Drive/最終レポート/bgimg');
    ds = imageDatastore(path, 'FileExtensions', '.jpg');
    list = ds.Files;
    OUTDIR1 = 'negImgDir_easy';
    mkdir(OUTDIR1);
    NegIdx = [1:size(list, 1)];
    rng(1);
    NegList = list(NegIdx(randperm(length(NegIdx), 200)));
    for j = 1:length(NegList)
        fnameNeg = strcat(OUTDIR1, '/', num2str(j, '%04d'), '.jpg');
        copyfile(NegList{j}, fnameNeg);
    end
    % neg(難しい方)
    list = textread('neg_urllist.txt', '%s');
    OUTDIR2 = 'negImgDir_diff';
    mkdir(OUTDIR2);
    for i = 1:size(list, 1)
        fname = strcat(OUTDIR2, '/', num2str(i, '%04d'), '.jpg');
        websave(fname, list{i});
    end
end