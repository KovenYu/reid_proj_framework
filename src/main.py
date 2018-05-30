from utils import *
from trainers import ReidTrainer
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    opts = BaseOptions()
    args = opts.parse()
    occupy_gpu_memory(args.gpu)
    logger = Logger(args.save_path)
    opts.print_options(logger)
    train_loader, gallery_loader, probe_loader = get_reid_dataloaders(args)
    num_classes = train_loader.dataset.return_num_class()

    if args.resume:
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, num_classes, logger)
        start_epoch = 0

    total_epoch = args.epochs

    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, total_epoch):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))

        trainer.train_epoch(train_loader, gallery_loader, probe_loader, epoch)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()


if __name__ == '__main__':
    main()
