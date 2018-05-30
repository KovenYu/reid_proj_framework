from resnet import resnet50
from utils import *
import torch.nn as nn
import torch
import os
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
        checkpoint = torch.load(args.pretrain_path)
        fixed_layers = ('fc',)
        state_dict = reset_state_dict(checkpoint, self.net, *fixed_layers)
        self.net.load_state_dict(state_dict)
        self.logger.print_log('loaded pre-trained feature net')

        self.cls_loss = nn.CrossEntropyLoss().cuda()

        bn_params, other_params = partition_params(self.net, 'bn')
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                         {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)

        recording_stats = ('acc/r1', 'mAP', 'loss_cls')
        self.recorder = Recorder(args.epochs, recording_stats)
        logger.print_log('recorder: observing stats: {}'.format(recording_stats))

    def train_epoch(self, train_loader, gallery_loader, probe_loader, epoch):
        adjust_learning_rate(self.optimizer, (self.args.lr,), epoch, self.args.epochs, self.args.lr_strategy)

        batch_time_meter = AverageMeter()
        stats = self.recorder.recording_stats
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()

        end = time.time()
        for i, train_tuple in enumerate(train_loader):
            imgs = train_tuple[0].cuda()
            labels = train_tuple[1].cuda()

            stats = self.update_net(imgs, labels)
            for k, v in stats.items():
                meters_trn[k].update(v, self.args.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(train_loader), freq) + create_stat_string(meters_trn) + time_string())

        self.recorder.update(epoch=epoch, is_train=True, meters=meters_trn)
        self.logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

        meters_val = self.eval_performance(gallery_loader, probe_loader)
        is_best = self.recorder.update(epoch=epoch, is_train=False, meters=meters_val)
        self.logger.print_log('  **Test**  ' + create_stat_string(meters_val))

        self.recorder.plot_curve(os.path.join(self.args.save_path, 'learning_curve.png'))
        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoint.pth.tar"), is_best=is_best)

    def update_net(self, imgs, labels):
        predictions = self.net(imgs)[1]
        loss = self.cls_loss(predictions, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = compute_accuracy(predictions, labels)
        stats = {'acc/r1': acc,
                 'loss_cls': loss.item()}
        return stats

    def eval_performance(self, gallery_loader, probe_loader):
        stats = self.recorder.recording_stats
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net,
                                                                           index_feature=0, require_views=True)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net,
                                                                     index_feature=0, require_views=True)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
        rank1 = CMC[0]
        meters_val['acc/r1'].update(rank1, 1)
        meters_val['mAP'].update(MAP, 1)
        return meters_val
