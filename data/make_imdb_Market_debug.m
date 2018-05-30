% run make_imdb_Market first, to obtain Market.mat

source = 'Market.mat';

load(source)

idx = [];

probe_unilabels = unique(probe_labels);
nunique = numel(probe_unilabels);
for i = 1:nunique
    this_idx = find(probe_unilabels(i) == probe_labels);
    this_idx = this_idx(1);
    idx = [idx, this_idx];
    probe_view_ref(probe_unilabels(i)) = probe_views(this_idx);
end
probe_data = probe_data(:, :, :, idx);
probe_views = probe_views(idx);
probe_labels = probe_labels(idx);

idx = [];

unilabels = unique(gallery_labels);
nunique = numel(unilabels);
for i = 1:nunique
    this_label = unilabels(i);
    this_idx = find(this_label == gallery_labels);
%     this_idx = this_idx(1);
    for j = 1:numel(this_idx)
        this_this_idx = this_idx(j);
        this_gallery_view = gallery_views(this_this_idx);
        if ~any(find(this_label == probe_unilabels))
            break
        end
        corre_probe_view = probe_view_ref(this_label);
        if this_gallery_view ~= corre_probe_view
            break
        end
    end
    idx = [idx, this_this_idx];
end
gallery_data = gallery_data(:, :, :, idx);
gallery_views = gallery_views(idx);
gallery_labels = gallery_labels(idx);

idx = [];

unilabels = unique(train_labels);
nunique = numel(unilabels);
for i = 1:nunique
    this_idx = find(unilabels(i) == train_labels);
    this_idx = this_idx(1);
    idx = [idx, this_idx];
end
train_data = train_data(:, :, :, idx);
train_views = train_views(idx);
train_labels = train_labels(idx);

save(['debug', source], 'gallery_data', 'gallery_labels', 'gallery_views', ...
    'probe_labels', 'probe_data', 'probe_views', ...
    'train_labels', 'train_views', 'train_data', '-v7.3')