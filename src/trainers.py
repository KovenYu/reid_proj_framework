from resnet import resnet50
from utils import *
import torch.nn as nn
import torch
import os
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()


class ReidTrainer(Trainer):
    def __init__(self, args, num_classes, logger):
        super(ReidTrainer, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.logger = logger

        self.net = resnet50(pretrained=False, num_classes=num_classes).cuda()
        if args.pretrain_path is None:
            self.logger.print_log('do not use pre-trained model. train from scratch.')
        elif os.path.isfile(args.pretrain_path):
            state_dict = torch.load(args.pretrain_path)
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.net.load_state_dict(state_dict, strict=False)
            self.logger.print_log('loaded pre-trained model from {}'.format(args.pretrain_path))
        else:
            self.logger.print_log('{} is not a file. train from scratch.'.format(args.pretrain_path))

        self.cls_loss = nn.CrossEntropyLoss().cuda()

        bn_params, other_params = partition_params(self.net, 'bn')
        fc_params, _ = partition_params(self.net, 'specified', 'fc')
        other_params_ = list(set(other_params) - set(fc_params))
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                          {'params': fc_params, 'weight_decay': 0},
                                         {'params': other_params_}], lr=args.lr, momentum=0.9, weight_decay=args.wd)

        self.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    def train_epoch(self, train_loader, epoch):
        adjust_learning_rate(self.optimizer, (self.args.lr,), epoch, self.args.epochs, self.args.lr_strategy)

        batch_time_meter = AverageMeter()
        stats = ('loss',)
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()

        end = time.time()
        for i, train_tuple in enumerate(train_loader):
            imgs = train_tuple[0].cuda()
            labels = train_tuple[1].cuda()

            predictions = self.net(imgs)[1]
            loss = self.cls_loss(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i == 0 and self.args.record_grad:
                name = 'conv1.weight'
                param = self.net.conv1.weight
                self.recorder.add_histogram(name+'_grad', param.grad, epoch, bins='auto')

            for k in stats:
                v = locals()[k]
                meters_trn[k].update(v, self.args.batch_size)
                self.recorder.add_scalar(k, meters_trn[k].avg, epoch)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(train_loader), freq) + create_stat_string(meters_trn) + time_string())

        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, gallery_loader, probe_loader, epoch):
        stats = ('r1',)
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, _ = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
        r1 = CMC[0]
        for k in stats:
            v = locals()[k]
            meters_val[k].update(v, self.args.batch_size)
            self.recorder.add_scalar(k, meters_val[k].avg, epoch)
        return meters_val
