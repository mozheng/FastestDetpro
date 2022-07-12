import argparse
import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from module.detector import Detector
from module.loss import DetectorLoss
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator
from utils.tool import *


def set_random_seed(seed, deterministic=False):
    """ Set random state to random libray, numpy, torch and cudnn.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def parse_args():
    """ Parse arguments.
    Returns:
        args: args object.
    """
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None,
                        help='.weight config')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dist_backend', type=str,
                        default='nccl', help='distributed backend')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8880')
    parser.add_argument('--local_rank', type=int,
                        default=0, help='distributed rank')
    parser.add_argument('--world_size', type=int, default=4,
                        help='distributed world size')
    parser.add_argument('--tflog', type=str, default='',
                        help='tensorboard log dir')
    parser.add_argument('--amp', action='store_true', default=False)
    return parser.parse_args()


def init(opt):
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    # 解析yaml配置文件
    opt.cfg = LoadYaml(opt.yaml)
    if opt.seed == 0:
        opt.seed = random.randint(1, 10000)
    set_random_seed(opt.seed, deterministic=True)
    return opt


def load_model(local_rank, opt):

    device = torch.device("cuda:%d" % local_rank)
    if opt.weight is not None:
        model = Detector(opt.cfg.category_num, True).to(device)
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(opt.weight).items()})
    else:
        model = Detector(opt.cfg.category_num, False).to(device)
    if local_rank == 0:
        # # 打印网络各层的张量维度
        summary(model, input_size=(3, opt.cfg.input_height, opt.cfg.input_width))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank])
    loss_function = DetectorLoss(device)

    # 数据集加载
    train_dataset = TensorDataset(
        opt.cfg.train_txt, opt.cfg.input_width, opt.cfg.input_height, True)
    val_dataset = TensorDataset(
        opt.cfg.val_txt, opt.cfg.input_width, opt.cfg.input_height, False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # 训练集
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.cfg.batch_size,
                                                   collate_fn=collate_fn,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   persistent_workers=True,
                                                   sampler=train_sampler
                                                   )
    # 验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.cfg.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 persistent_workers=True,
                                                 sampler=val_sampler
                                                 )
    return model, loss_function, train_dataloader, val_dataloader


def train(local_rank, opt):

    if local_rank == 0:
        # TFboard
        if opt.tflog != '':
            tfwriter = SummaryWriter(opt.tflog)
        else:
            tfwriter = SummaryWriter()

    dist.init_process_group(backend="nccl" if (dist.is_nccl_available() and opt.dist_backend == "nccl") else "gloo",
                            init_method=opt.dist_url, rank=local_rank, world_size=opt.world_size)

    model, loss_function, train_dataloader, val_dataloader = load_model(
        local_rank, opt)

    device = torch.device("cuda:%d" % local_rank)

    # 构建优化器
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opt.cfg.learn_rate,
                          momentum=0.949,
                          weight_decay=0.0005,
                          )
    # 学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=opt.cfg.milestones,
                                               gamma=0.1)
    # 迭代训练
    batch_num = 0
    # print('Starting training for %g epochs...' % cfg.end_epoch)
    for epoch in range(opt.cfg.start_epoch, opt.cfg.end_epoch + 1):
        model.train()
        pbar = tqdm(train_dataloader)
        for imgs, targets in pbar:
            # 数据预处理
            # torch.distributed.barrier()
            imgs = imgs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()
            if opt.amp:
                # 模型推理
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    # loss计算
                    iou, obj, cls, total = loss_function(preds, targets)
                # 反向传播求解梯度
                scaler.scale(total).backward()
                # 更新模型参数
                scaler.step(optimizer)
                scaler.update()
            else:
                # 模型推理
                preds = model(imgs)
                # loss计算
                iou, obj, cls, total = self.loss_function(preds, targets)
                # 反向传播求解梯度
                total.backward()
                # 更新模型参数
                optimizer.step()

            # 学习率预热
            for g in optimizer.param_groups:
                warmup_num = 5 * len(train_dataloader)
                if batch_num <= warmup_num and opt.cfg.start_epoch == 0:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = opt.cfg.learn_rate * scale
                lr = g["lr"]
            if local_rank == 0:
                # 打印相关训练信息
                tfwriter.add_scalar('train/lr', lr, global_step=epoch)
                tfwriter.add_scalar(
                    'train/iou_loss', iou, global_step=epoch)
                tfwriter.add_scalar(
                    'train/obj_loss', obj, global_step=epoch)
                tfwriter.add_scalar(
                    'train/cls_loss', cls, global_step=epoch)
                tfwriter.add_scalar('train/total_loss',
                                    total, global_step=epoch)
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou, obj, cls, total)
                pbar.set_description(info)
            batch_num += 1

        # 模型验证及保存
        if epoch % 10 == 0 and epoch > opt.cfg.start_epoch:
            if local_rank == 0:
                # 模型评估
                model.eval()
                print("computer mAP...")
                evaluation = CocoDetectionEvaluator(opt.cfg.names, device)
                mAP05 = evaluation.compute_map(val_dataloader, model)
                tfwriter.add_scalar('eval/mAP50', mAP05, global_step=epoch)
                torch.save(
                    model.state_dict(), "checkpoint/weight_AP05:%f_%d-epoch.pth" % (mAP05, epoch))

        # 学习率调整
        scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    opt = init(args)
    mp.spawn(train, nprocs=opt.world_size, args=(opt,))
