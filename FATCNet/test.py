import logging
import random
import sys
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets_n.dataset_synapse1 import Synapse_dataset
from networks.vit_seg_modeling import FATCNet as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from util.metric import ROCMetric, mIoU, PD_FA
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse/test_npz_other', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True,action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./result', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

parser.add_argument('--method', type=str, default='gradcam',
                    choices=['gradcam', 'gradcam++',
                             'scorecam', 'xgradcam',
                             'ablationcam', 'eigencam',
                             'eigengradcam', 'layercam'],
                    help='Can be gradcam/gradcam++/scorecam/xgradcam'
                         '/ablationcam/eigencam/eigengradcam/layercam')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    mioU = mIoU(1)
    ROC = ROCMetric(1, 10)
    Pd_Fa = PD_FA(1, 10)
    d = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image, label = image.squeeze(0).cpu().detach(), label.squeeze(0).cpu().detach()
        input = image.unsqueeze(0).float().cuda()

        with torch.no_grad():
            net_1 = net(input)
            # preds = torch.argmax(torch.softmax(net_1, dim=1), dim=1).squeeze(0).cpu().detach()
            proc = torch.softmax(net_1, dim=1)[:, 1].cpu().detach()
            preds = net_1[:, 1, :, :].squeeze(0).cpu().detach()

            nIoU = mIoU(1)
            ROC.update(proc, label)
            Pd_Fa.update(preds, label)
            mioU.update(preds, label)
            nIoU.update(preds, label)
            _, n_IOU = nIoU.get()
            d.append(n_IOU)
            # _, mean_IOU = mioU.get()

            predsss = np.array(preds > 0).astype('int64') * 255
            predsss = np.uint8(predsss)
            labelsss = label * 255
            labelsss = np.uint8(labelsss)

            img = Image.fromarray(predsss.reshape(args.img_size, args.img_size))
            mask = Image.fromarray(labelsss.reshape(args.img_size, args.img_size))

            print(test_save_path + '/' + case_name + '.png')
            img.save(test_save_path + '/' + case_name + '.png')
            mask.save(test_save_path + '/' + case_name + '_GT' + '.png')


            # print(mean_IOU)

    _, mean_IOU = mioU.get()
    tp_rates, fp_rates, recall, precision, F1_score = ROC.get()
    FA, PD = Pd_Fa.get(len(db_test))
    logging.info('Testing performance in best val model: mIou : %f ' % (mean_IOU))
    logging.info('Testing performance in best val model: nIou : %f ' % (sum(d) / (len(d))))
    logging.info('Testing performance in best val model: F1 : %f ' % F1_score)
    logging.info('Testing performance in best val model: FA : %f ' % (FA[0] * 1000000))
    logging.info('Testing performance in best val model: FD : %f ' % PD[0])
    print('mIoU: {:.4f}  nIoU: {:.4f}  F1: {:.4f} FA:{:.4f} PD:{:.4f}'.format(
        float(mean_IOU), float(sum(d) / (len(d))), F1_score, FA[0] * 1000000, PD[0]))

    print('tpr:', tp_rates)
    print('fpr:', fp_rates)
    print('recall:', recall)
    print('precision:', precision)

    return "Testing Finished!"


def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            # 'volume_path': '/home/yj/train_data/train/IRSTD1k_npz',
            # 'list_dir': '/home/yj/train_data/list/IRSTD1k',
            # 'volume_path': '/home/yj/train_data/train/NUAA_npz',
            # 'list_dir': '/home/yj/train_data/list/NUAA/82',
            'volume_path': '/home/yj/train_data/train/NUDT_npz',
            'list_dir': '/home/yj/train_data/list/NUDT',
            # 'volume_path': '/home/yj/train_data/train/BSIRST_v2_npz',
            # 'list_dir': '/home/yj/train_data/list/BSIRST_v2',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    print(net)

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    # snapshot = '/home/yj/FATCNet/model/IRSDT-1k/epoch_best.pth'
    # snapshot = '/home/yj/FATCNet/model/NUAA/epoch_best.pth'
    snapshot = '/home/yj/FATCNet/model/NUDT/epoch_best.pth'
    # snapshot = '/home/yj/FATCNet/model/BSIRST_v2/epoch_best.pth'

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]


    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = 'result/v2_END'
        # test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)