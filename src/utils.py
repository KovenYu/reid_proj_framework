import os, sys, time
import numpy as np
import matplotlib
import subprocess
import torch
import argparse
import torchvision.transforms as transforms
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


def extract_features(loader, model, index_feature=1, require_views=True):
    """
    extract features for the given loader using the given model
    :param loader: must return (imgs, labels, views)
    :param model: returns a tuple, containing the feature
    :param index_feature: in the tuple returned by model, the index of the feature
    :param require_views: if True, also return view information
    :return: features, labels, (if required) views, all as n-by-d numpy array
    """
    # switch to evaluate mode
    model.eval()

    labels = torch.zeros((len(loader.dataset),), dtype=torch.long)
    views = torch.zeros((len(loader.dataset),), dtype=torch.long)

    idx = 0
    assert loader.dataset.require_views == require_views, 'require_views not consistent in loader and specified option'
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        with torch.no_grad():
            output_tuple = model(imgs)
        feature_batch = output_tuple[index_feature]
        feature_batch = feature_batch.data.cpu()

        if i == 0:
            feature_dim = feature_batch.shape[1]
            features = torch.zeros((len(loader.dataset), feature_dim))

        batch_size = label_batch.shape[0]
        features[idx: idx + batch_size, :] = feature_batch
        labels[idx: idx + batch_size] = label_batch
        if require_views:
            view_batch = data[2]
            views[idx: idx + batch_size] = view_batch
        idx += batch_size

    features_np = features.numpy()
    labels_np = labels.numpy()
    views_np = views.numpy()
    if require_views:
        return features_np, labels_np, views_np
    else:
        return features_np, labels_np


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
    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
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


def get_transfer_dataloaders(args):
    """
    get source/target/gallery/probe dataloaders, where
    source loader is FullTraining, others are Market.
    :param args: using args.source, args.target, args.img_size, args.crop_size, args.padding, args.batch_size
    args.use_source_stat: if True, using mean and std from source_data to normalize target/gal/prb data;
    otherwise, using mean and std from target_data
    :return: the four dataloaders
    """

    source_data = FullTraining('data/{}.mat'.format(args.source))
    target_data = Market('data/{}.mat'.format(args.target), state='train')
    gallery_data = Market('data/{}.mat'.format(args.target), state='gallery')
    probe_data = Market('data/{}.mat'.format(args.target), state='probe')

    mean = source_data.return_mean() / 255.0
    std = source_data.return_std() / 255.0

    source_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(args.img_size),
         transforms.RandomCrop(args.crop_size, args.padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if not args.use_source_stat:
        mean = target_data.return_mean() / 255.0
        std = target_data.return_mean() / 255.0
    target_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(args.img_size),
         transforms.RandomCrop(args.crop_size, args.padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    source_data.turn_on_transform(transform=source_transform)
    target_data.turn_on_transform(transform=target_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    source_loader = torch.utils.data.DataLoader(source_data, batch_size=args.batch_size, shuffle=True,
                                                num_workers=10, pin_memory=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=True,
                                                num_workers=10, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=10, pin_memory=True)

    return source_loader, target_loader, gallery_loader, probe_loader


def get_reid_dataloaders(args):
    """
    get train/gallery/probe dataloaders.
    :param args: using args.dataset, args.img_size, args.crop_size, args.padding, args.batch_size
    :return:
    """

    train_data = Market('data/{}.mat'.format(args.dataset), state='train')
    gallery_data = Market('data/{}.mat'.format(args.dataset), state='gallery')
    probe_data = Market('data/{}.mat'.format(args.dataset), state='probe')

    mean = train_data.return_mean() / 255.0
    std = train_data.return_std() / 255.0

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(args.img_size),
         transforms.RandomCrop(args.crop_size, args.padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=10, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=10, pin_memory=True)

    return train_loader, gallery_loader, probe_loader


def test():
    opts = BaseOptions()
    args = opts.parse()
    print(args)


if __name__ == '__main__':
    test()
