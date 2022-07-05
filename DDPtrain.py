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
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8880')
    parser.add_argument('--local_rank', type=int, default=0, help='distributed rank')
    parser.add_argument('--world_size', type=int, default=1, help='distributed world size')
    parser.add_argument('--tflog', type=str, default='', help='tensorboard log dir')
    return parser.parse_args()


class FastestDet:
    def __init__(self, opt) -> None:
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
        # 解析yaml配置文件
        self.cfg = LoadYaml(opt.yaml)    
        print(self.cfg) 
        if opt.seed == 0:
            opt.seed = random.randint(1, 10000)
        # 初始化模型结构
        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model = Detector(self.cfg.category_num, True)
            self.model.load_state_dict(torch.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False)
        # # 打印网络各层的张量维度
        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

        # TFboard
        if opt.tflog != '':
            self.tfwriter = SummaryWriter(opt.tflog)
        else:
            self.tfwriter = SummaryWriter()

        #构建优化器
        print("use SGD optimizer")
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.cfg.learn_rate,
                                   momentum=0.949,
                                   weight_decay=0.0005,
                                   )
        # 学习率衰减策略
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.cfg.milestones,
                                                        gamma=0.1)

        # 数据集加载
        val_dataset = TensorDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False)
        train_dataset = TensorDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        #验证集
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.cfg.batch_size,
                                                          shuffle=False,
                                                          collate_fn=collate_fn,
                                                          num_workers=4,
                                                          drop_last=False,
                                                          persistent_workers=True,
                                                          sampler=val_sampler
                                                          )
        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.cfg.batch_size,
                                                            collate_fn=collate_fn,
                                                            num_workers=4,
                                                            drop_last=True,
                                                            persistent_workers=True,
                                                            sampler=train_sampler
                                                            )
    def load_model(self, local_rank):
        model = self.model.copy()
        device = torch.device("cuda:%d" % local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        loss_function = DetectorLoss(device)
        return model, loss_function


    def DDPtrain(self):
        mp.spawn(self.train, nprocs=self.opt.world_size, args=())
        dist.destroy_process_group()

    def train(self, local_rank):
        set_random_seed(self.opt.seed, deterministic=True)
        dist.init_process_group(backend="nccl" if (dist.is_nccl_available() and self.opt.dist_backend=="nccl") else "gloo",
                            init_method=self.opt.dist_url, rank=local_rank, world_size=self.opt.world_size)
        
        model, loss_function = self.load_model(local_rank)

        device = torch.device("cuda:%d" % local_rank)

        # 迭代训练
        batch_num = 0
        # print('Starting training for %g epochs...' % self.cfg.end_epoch)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.cfg.end_epoch + 1):
            model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                # 数据预处理
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)

                self.optimizer.zero_grad()
                # 模型推理
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    # loss计算
                    iou, obj, cls, total = loss_function(preds, targets)
                # 反向传播求解梯度
                scaler.scale(total).backward()
                # 更新模型参数
                scaler.step(self.optimizer)
                scaler.update()

                # 学习率预热
                for g in self.optimizer.param_groups:
                    warmup_num =  5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num/warmup_num, 4)
                        g['lr'] = self.cfg.learn_rate * scale
                    lr = g["lr"]
                if local_rank == 0:
                    # 打印相关训练信息
                    torch.distributed.barrier()
                    self.tfwriter.add_scalar('train/lr', lr, global_step=epoch)
                    self.tfwriter.add_scalar('train/iou_loss', iou, global_step=epoch)
                    self.tfwriter.add_scalar('train/obj_loss', obj, global_step=epoch)
                    self.tfwriter.add_scalar('train/cls_loss', cls, global_step=epoch)
                    self.tfwriter.add_scalar('train/total_loss', total, global_step=epoch)
                    info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                        epoch, lr, iou, obj, cls, total)
                    pbar.set_description(info)
                batch_num += 1

            # 模型验证及保存
            if epoch % 10 == 0 and epoch > 0:
                if local_rank == 0:
                    torch.distributed.barrier()
                    # 模型评估
                    model.eval()
                    print("computer mAP...")
                    evaluation = CocoDetectionEvaluator(self.cfg.names, device)
                    mAP05 = evaluation.compute_map(self.val_dataloader, model)
                    self.tfwriter.add_scalar('eval/mAP50', mAP05, global_step=epoch)
                    torch.save(model.state_dict(), "checkpoint/weight_AP05:%f_%d-epoch.pth"%(mAP05, epoch))

            # 学习率调整
            self.scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    fd = FastestDet()
    fd.DDPtrain()
    