import os, sys, time
import numpy as np
import matplotlib
import subprocess
import torch
import argparse
import torchvision.transforms as transforms
import torchvision
from random import sample
matplotlib.use('agg')
from ReIDdatasets import FullTraining, Market
import torch.cuda as cutorch
import yaml
from tensorboardX import SummaryWriter


class BaseOptions(object):
    """
    base options for deep learning for Re-ID.
    parse basic arguments by parse(), print all the arguments by print_options()
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = None

        self.parser.add_argument('--save_path', type=str, default='debug', help='Folder to save checkpoints and log.')
        self.parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--gpu', type=str, default='0', help='gpu used.')

    def parse(self):
        self.args = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        with open(os.path.join(self.args.save_path, 'args.yaml')) as f:
            extra_args = yaml.load(f)
        self.args = argparse.Namespace(**vars(self.args), **extra_args)
        return self.args

    def print_options(self, logger):
        logger.print_log("")
        logger.print_log("----- options -----".center(120, '-'))
        args = vars(self.args)
        string = ''
        for i, (k, v) in enumerate(sorted(args.items())):
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(args.items())-1:
                logger.print_log(string)
                string = ''
        logger.print_log("".center(120, '-'))
        logger.print_log("")


class Logger(object):
    def __init__(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_log(self, string):
        self.file.write("{}\n".format(string))
        self.file.flush()
        print(string)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def extract_features(loader, model, index_feature=None):
    """
    extract features for the given loader using the given model
    if loader.dataset.require_views is False, the returned 'views' are empty.
    :param loader: a ReIDDataset that has attribute require_views
    :param model: returns a tuple containing the feature or only return the feature. if latter, index_feature be None
    model can also be a tuple of nn.Module, indicating that the feature extraction is multi-stage.
    in this case, index_feature should be a tuple of the same size.
    :param index_feature: in the tuple returned by model, the index of the feature.
    if the model only returns feature, this should be set to None.
    :return: features, labels, views; feature is torch.float; labels and views are torch.long
    ALL are in cpu().
    """
    if type(model) is not tuple:
        models = (model,)
        indices_feature = (index_feature,)
    else:
        assert len(model) == len(index_feature)
        models = model
        indices_feature = index_feature
    for m in models:
        m.eval()

    labels = torch.tensor([], dtype=torch.long)
    views = torch.tensor([], dtype=torch.long)
    features = torch.tensor([], dtype=torch.float)

    require_views = loader.dataset.require_views
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        inputs = imgs
        for m, feat_idx in zip(models, indices_feature):
            with torch.no_grad():
                output_tuple = m(inputs)
            feature_batch = output_tuple if feat_idx is None else output_tuple[feat_idx]
            inputs = feature_batch
        feature_batch = feature_batch.detach().cpu()

        features = torch.cat((features, feature_batch), dim=0)
        labels = torch.cat((labels, label_batch), dim=0)
        if require_views:
            view_batch = data[2]
            views = torch.cat((views, view_batch), dim=0)
    return features, labels, views


def create_stat_string(meters):
    stat_string = ''
    for stat, meter in meters.items():
        stat_string += '{} {:.3f}   '.format(stat, meter.avg)
    return stat_string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    """
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)
    
    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[:, i] = cmc_

        num_correct = positions_correct.shape[0]
        for j in range(num_correct):
            last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
            current_precision = float(j+1) / float(positions_correct[j]+1)
            ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC*100, MAP*100


def occupy_gpu_memory(gpu_id, maximum_usage=None, buffer_memory=2000):
    """
    As pytorch is dynamic, you might wanna take enough GPU memory to avoid OOM when you run your code
    in a messy server.
    if maximum_usage is specified, this function will return a dummy buffer which takes memory of
    (current_available_memory - (maximum_usage - current_usage) - buffer_memory) MB.
    otherwise, maximum_usage would be replaced by maximum usage till now, which is returned by
    torch.cuda.max_memory_cached()
    :param gpu_id:
    :param maximum_usage: float, measured in MB
    :param buffer_memory: float, measured in MB
    :return:
    """
    gpu_id = int(gpu_id)
    if maximum_usage is None:
        maximum_usage = cutorch.max_memory_cached()
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split(b'\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    available_memory = gpu_memory_map[gpu_id]
    if available_memory < buffer_memory+1000:
        print('Gpu memory has been mostly occupied (although maybe not by you)!')
    else:
        memory_to_occupy = int((available_memory - (maximum_usage - cutorch.memory_cached()) - buffer_memory))
        dim = int(memory_to_occupy * 1024 * 1024 * 8 / 32)
        x = torch.zeros(dim, dtype=torch.int)
        x.pin_memory()
        print('Occupied {}MB extra gpu memory.'.format(memory_to_occupy))
        x_ = x.cuda()
        del x_


def reset_state_dict(state_dict, model, *fixed_layers):
    """
    :param state_dict: to be modified
    :param model: must be initialized
    :param fixed_layers: must be in both state_dict and model.
    :return:
    """
    for k in state_dict.keys():
        for l in fixed_layers:
            if k.startswith(l):
                if k.endswith('bias'):
                    state_dict[k] = model.__getattr__(l).bias.data
                elif k.endswith('weight'):
                    state_dict[k] = model.__getattr__(l).weight.data
                elif k.endswith('running_mean'):
                    state_dict[k] = model.__getattr__(l).running_mean
                elif k.endswith('running_var'):
                    state_dict[k] = model.__getattr__(l).running_var
                else:
                    assert False, 'Not specified param type: {}'.format(k)
    return state_dict


def save_checkpoint(trainer, epoch, save_path, is_best=False):
    logger = trainer.logger
    recorder = trainer.recorder
    trainer.logger = None
    trainer.recorder = None
    if not os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    torch.save((trainer, epoch), save_path)
    if is_best:
        best_path = save_path + '.best'
        torch.save((trainer, epoch), best_path)
    trainer.logger = logger
    trainer.recorder = recorder


def load_checkpoint(args, logger):
    """
    load a checkpoint (containing a trainer and an epoch number) and assign a logger to the loaded trainer.
    the differences between the loaded trainer.args and input args would be print to logger.
    :param args:
    :param logger:
    :return:
    """
    load_path = args.resume
    assert os.path.isfile(load_path)
    logger.print_log("=> loading checkpoint '{}'".format(load_path))
    (trainer, epoch) = torch.load(load_path)
    trainer.logger = logger
    trainer.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    old_args = trainer.args
    trainer.args = args

    attributes = vars(args)
    old_attributes = vars(old_args)
    for name, value in attributes.items():
        if name == 'resume' or name == 'gpu':
            continue
        old_value = old_attributes[name]
        if old_value != value:
            logger.print_log("args.{} was {} but now is replaced by the newly specified one: {}.".format(name, value,
                                                                                                         old_value))

    return trainer, epoch


def adjust_learning_rate(optimizer, init_lr, epoch, total_epoch, strategy, lr_list=None):
    """
    :param optimizer:
    :param init_lr: tuple of float, either has one element (for all param_groups in optimizer)
    or has len(param_groups) elements. If latter, each element corresponds to a param_group
    :param epoch: int, current epoch index
    :param total_epoch: int
    :param strategy: choices are: ['constant', 'resnet_style', 'cyclegan_style', 'specified'],
    'constant': keep learning rate unchanged through training
    'resnet_style': divide lr by 10 in total_epoch/2, by 100 in total_epoch*(3/4)
    'cyclegan_style': linearly decrease lr to 0 after total_epoch/2
    'specified': according to the given lr_list
    :param lr_list: numpy array, shape=(n_groups, total_epoch), only required when strategy == 'specified'
    :return: new_lr, tuple, has the same shape as init_lr
    """

    n_group = len(optimizer.param_groups)
    if len(init_lr) == 1:
        init_lr = tuple([init_lr for _ in range(n_group)])
    else:
        assert len(init_lr) == n_group
    if strategy == 'constant':
        new_lr = init_lr
        return new_lr
    elif strategy == 'resnet_style':
        lr_list = np.ones((n_group, total_epoch), dtype=float)
        for i in range(n_group):
            lr_list[i, :] *= init_lr[i]
        factors = np.ones(total_epoch,)
        factors[int(total_epoch/2):] *= 0.1
        factors[int(3*total_epoch/4):] *= 0.1
        lr_list *= factors
        new_lr = lr_list[:, epoch]
    elif strategy == 'cyclegan_style':
        lr_list = np.ones((n_group, total_epoch), dtype=float)
        for i in range(n_group):
            lr_list[i, :] *= init_lr[i]
        factors = np.ones(total_epoch,)
        n_elements = len(factors[int(total_epoch/2):])
        factors[int(total_epoch / 2):] = np.linspace(1, 0, n_elements)
        lr_list *= factors
        new_lr = lr_list[:, epoch]
    elif strategy == 'specified':
        assert lr_list is not None, 'if strategy is "specified", must provide lr_list'
        new_lr = lr_list[:, epoch]
    else:
        assert False, 'unknown strategy: {}'.format(strategy)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr[i]
    return tuple(new_lr)


def compute_accuracy(predictions, labels):
    """
    compute classification accuracy, measured by percentage.
    :param predictions: tensor. size = N*d
    :param labels: tensor. size = N
    :return: python number, the computed accuracy
    """
    predicted_labels = torch.argmax(predictions, dim=1)
    n_correct = torch.sum(predicted_labels == labels).item()
    batch_size = torch.numel(labels)
    acc = float(n_correct) / float(batch_size)
    return acc * 100


def partition_params(module, strategy, *desired_modules):
    """
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: tuple of strings, each corresponds to a specific module
    :return: two iterators
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))
    else:
        assert False, 'unknown strategy: {}'.format(strategy)
    all_params_set = set(module.parameters())
    other_params_set = all_params_set.difference(desired_params_set)
    desired_params = (p for p in desired_params_set)
    other_params = (p for p in other_params_set)
    return desired_params, other_params


def get_transfer_dataloaders(source, target, img_size, crop_size, padding, batch_size, use_source_stat=False):
    """
    get source/target/gallery/probe dataloaders, where
    source loader is FullTraining, others are Market.
    :return: the four dataloaders
    """

    source_data = FullTraining('data/{}.mat'.format(source))
    target_data = Market('data/{}.mat'.format(target), state='train')
    gallery_data = Market('data/{}.mat'.format(target), state='gallery')
    probe_data = Market('data/{}.mat'.format(target), state='probe')

    mean = source_data.return_mean() / 255.0
    std = source_data.return_std() / 255.0

    source_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if not use_source_stat:
        mean = target_data.return_mean() / 255.0
        std = target_data.return_mean() / 255.0
    target_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    source_data.turn_on_transform(transform=source_transform)
    target_data.turn_on_transform(transform=target_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    source_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=True,
                                                num_workers=10, pin_memory=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=True,
                                                num_workers=10, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=10, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=batch_size, shuffle=False,
                                               num_workers=10, pin_memory=True)

    return source_loader, target_loader, gallery_loader, probe_loader


def get_reid_dataloaders(dataset, img_size, crop_size, padding, batch_size):
    """
    get train/gallery/probe dataloaders.
    :return:
    """

    train_data = Market('data/{}.mat'.format(dataset), state='train')
    gallery_data = Market('data/{}.mat'.format(dataset), state='gallery')
    probe_data = Market('data/{}.mat'.format(dataset), state='probe')

    mean = train_data.return_mean() / 255.0
    std = train_data.return_std() / 255.0

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=10, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=10, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=batch_size, shuffle=False,
                                               num_workers=10, pin_memory=True)

    return train_loader, gallery_loader, probe_loader


def compare_grad(params, losses, recorder, step):
    """
    compare gradient magnitudes of several losses, using a tensorboard histogram
    :param params: dict of parameters to observe
    :param losses: dict of losses to observe
    :param recorder: tensorboardX.SummaryWriter
    :param step: global step
    :return:
    """
    for param_name in params:
        param = params[param_name]
        for loss_name in losses:
            loss = losses[loss_name]
            grad = torch.autograd.grad(loss, param, grad_outputs=torch.tensor(1.0).cuda(), retain_graph=True,
                                       allow_unused=True)[0]
            if grad is not None:
                recorder.add_histogram("{}_{}".format(param_name, loss_name), grad, step, bins='auto')


def find_wrong_match(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
    find the probe samples which result in a wrong match at rank-1.
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    prb_idx: list of int, length == n_found_wrong_prb
    gal_idx: list of np array, each of which associating with the element in prb_idx
    correct_indicators: list of np array corresponding to gal_idx, indicating whether that gal is a correct match.
    """
    is_view_sensitive = False
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        is_view_sensitive = True
    prb_idx = []
    gal_idx = []
    correct_indicators = []

    for i in range(num_probe):
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        if pos_first_correct != 0:
            prb_idx.append(i)
            gal_idx.append(ranking_list)
            correct_indicators.append(probe_label == inference_list)

    return prb_idx, gal_idx, correct_indicators


def plot_ranking_imgs(gal_dataset, prb_dataset, gal_idx, prb_idx, n_gal=8, n_prb=8, size=(224, 224), save_path='',
                      correct_indicators=None):
    """
    plot ranking imgs and save it.
    :param gal_dataset: should support indexing and return a tuple, in which the first element is an img,
           represented as np array
    :param prb_dataset:
    :param gal_idx: list of np.array, each of which corresponds to the element in prb_idx
    :param prb_idx: list of int, indexing the prb_dataset
    :param n_gal: number of gallery imgs shown in a row (for a probe).
    :param n_prb: number of probe imgs shown, i.e. the number of rows. randomly chosen in the given list.
    :param size: resize all shown imgs
    :param save_path: directory to save; the file name is ranking_(time string).png
    :param correct_indicators: list of np array corresponding to gal_idx, indicating whether that
           gal is a correct match. if specified, each correct match will has a small green box in the upper-left.
    :return:
    """
    assert len(prb_idx) == len(gal_idx)
    if correct_indicators is not None:
        assert len(prb_idx) == len(correct_indicators)
    box_size = tuple(map(lambda x: int(x/12.0), size))

    is_gal_on = gal_dataset.on_transform
    is_prb_on = prb_dataset.on_transform
    gal_dataset.turn_off_transform()
    prb_dataset.turn_off_transform()

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    n_prb = len(prb_idx) if n_prb > len(prb_idx) else n_prb
    if correct_indicators is None:
        used = sample(list(zip(prb_idx, gal_idx)), n_prb)
        imgs = []
        for p_idx, g_idx_array in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx in g_idx_array[:n_gal_used]:
                gal_img = transform(gal_dataset[g_idx][0])
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))
    else:
        used = sample(list(zip(prb_idx, gal_idx, correct_indicators)), n_prb)
        imgs = []
        for p_idx, g_idx_array, correct_ind in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx, is_correct_match in zip(g_idx_array[:n_gal_used], correct_ind[:n_gal_used]):
                gal_img = transform(gal_dataset[g_idx][0])
                if is_correct_match:
                    gal_img[0, :box_size[0], :box_size[1]].zero_()
                    gal_img[1, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                else:
                    gal_img[0, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[1, :box_size[0], :box_size[1]].zero_()
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))

    filename = os.path.join(save_path, 'ranking_{}.png'.format(time_string()))
    torchvision.utils.save_image(imgs, filename, nrow=n_gal+1)
    print('saved ranking images into {}'.format(filename))
    gal_dataset.on_transform = is_gal_on
    prb_dataset.on_transform = is_prb_on


def test():
    opts = BaseOptions()
    args = opts.parse()
    print(args)


if __name__ == '__main__':
    test()
