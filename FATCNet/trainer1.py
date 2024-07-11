import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F
from util.metric import mIoU, ROCMetric, SigmoidMetric, SamplewiseSigmoidMetric
from util.utils import save_model
from util.scheduler import PolyLR
from util.drawing import drawing_loss, drawing_iou, drawing_f1
from datasets_n.dataset_synapse1 import Synapse_dataset, RandomGenerator


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # F.cross_entropy(x,y)工作过程就是(Log_Softmax+NllLoss)：①对x做softmax,使其满足归一化要求，结果记为x_soft;②对x_soft做对数运算
        # 并取相反数，记为x_soft_log;③对y进行one-hot编码，编码后与x_soft_log进行点乘，只有元素为1的位置有值而且乘的是1，
        # 所以点乘后结果还是x_soft_log
        # 总之，F.cross_entropy(x,y)对应的数学公式就是CE(pt)=-1*log(pt)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # pt是预测该类别的概率，要明白F.cross_entropy工作过程就能够理解
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_test = Synapse_dataset(base_dir=args.root_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn) #generator=torch.Generator(device = 'cuda')
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    # 损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    Focal_loss = FocalLoss()

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = PolyLR(optimizer=optimizer, base_lr=base_lr, num_epochs=args.max_epochs, policy='PolyLR',
                       warmup='linear', power=0.9, min_lr=0.00001, warmup_epochs=5)

    print(args.max_epochs)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    Eiter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # 评价指标
    ## evaluation metrics
    Iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    ROC = ROCMetric(1, 10)
    mioU = mIoU(1)

    best_miou = 0
    best_nIoU = 0
    best_IoU = 0

    # 画图
    dnum_epoch = []
    dtrain_loss = []
    dtest_loss = []
    dmIoU = []
    dnIoU = []
    dF1_score=[]

    for epoch_num in range(0, max_epoch):
        # 训练
        iterator_train = tqdm(trainloader, ncols=150)
        model.train()
        sumloss = 0
        scheduler.step(epoch_num)
        for i_batch, sampled_batch in enumerate(iterator_train):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)  #, gt_pre


            # loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_Focal = Focal_loss(outputs, label_batch.long())

            loss = 0.5 * loss_Focal + 0.5 * loss_dice

            sumloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1

            iterator_train.set_description('Epoch %d lr_ %f: loss : %f,loss_Focal : %f,loss_dice : %f' % (
            epoch_num, optimizer.param_groups[0]['lr'], loss, loss_Focal, loss_dice))

        train_loss = sumloss / len(iterator_train)

        dnum_epoch.append(epoch_num)
        dtrain_loss.append(train_loss)

        # 测试
        iterator_test = tqdm(testloader, ncols=150)
        sumloss = 0
        mioU.reset()
        Iou_metric.reset()
        nIoU_metric.reset()
        # 开始评估模式
        model.eval()
        # torch.no_grad() 用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(iterator_test):
                image, label = sampled_batch["image"], sampled_batch["label"]
                image, label = image.cuda(), label.cuda()

                Preds = model(image)


                loss_dice = dice_loss(Preds, label, softmax=True)
                loss_Focal = Focal_loss(Preds, label.long())

                loss = 0.5 * loss_Focal + 0.5 * loss_dice
                sumloss = loss.item() + sumloss

                # preds = torch.argmax(torch.softmax(Preds, dim=1), dim=1).cpu().detach()
                proc = torch.softmax(Preds, dim=1)[:, 1].cpu()
                preds = Preds[:, 1, :, :].cpu()
                labels = label.cpu()

                ROC.update(proc, labels)
                mioU.update(preds, labels)
                Iou_metric.update(preds, labels)
                nIoU_metric.update(preds, labels)
                _, IoU = Iou_metric.get()
                _, nIoU = nIoU_metric.get()
                _, mean_IOU = mioU.get()
                ture_positive_rate, false_positive_rate, recall, precision, f1_score = ROC.get()


                iterator_test.set_description('Epoch %d, loss %.4f, mean_IoU: %.7f' % (epoch_num, loss, mean_IOU))
                # 保存日志
                Eiter_num = Eiter_num + 1
                # logging.info('Test_iteration %d : loss : %f,mean_IOU : %f' % (Eiter_num, loss.item(), mean_IOU))

            test_loss = sumloss / len(iterator_test)
            dtest_loss.append(test_loss)
            dmIoU.append(mean_IOU)
            dnIoU.append(nIoU)
            dF1_score.append(f1_score)
            # logging.info('Epoch_num %d : testloss, : %f' % (epoch_num, testloss))
            best_miou, best_nIoU, best_IoU = save_model(mean_IOU, nIoU, IoU, best_miou, best_nIoU, best_IoU, snapshot_path,
                           train_loss, test_loss, recall, precision, epoch_num, model.state_dict())
            print(best_miou, best_nIoU, best_IoU)

            # 画图
            tdnum_epoch = torch.tensor(dnum_epoch, device='cpu')
            tdtrain_loss = torch.tensor(dtrain_loss, device='cpu')
            tdtest_loss = torch.tensor(dtest_loss, device='cpu')
            tdmIoU = torch.tensor(dmIoU, device='cpu')
            tdnIoU = torch.tensor(dnIoU, device='cpu')
            tdF1_score = torch.tensor(dF1_score, device='cpu')

            drawing_loss(tdnum_epoch, tdtrain_loss, tdtest_loss, snapshot_path)
            drawing_iou(tdnum_epoch, tdmIoU, tdnIoU, snapshot_path)
            drawing_f1(tdnum_epoch, tdF1_score, snapshot_path)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator_train.close()
            logging.info("save best_miou to {}".format(best_miou))
            logging.info("save best_miou to {}".format(best_nIoU))
            logging.info("save best_miou to {}".format(best_IoU))
            break
    writer.close()
    return "Training Finished!"