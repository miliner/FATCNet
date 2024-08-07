# @Time    : 2022/9/14 19:57
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : drawing.py
# @Software: PyCharm
import matplotlib.pyplot as plt


def drawing_loss(num_epoch, train_loss, test_loss, save_dir):
    plt.figure()
    plt.plot(num_epoch, train_loss, label='train_loss')
    plt.plot(num_epoch, test_loss, label='test_loss')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(save_dir + '/' + "fig_loss.png")


def drawing_iou(num_epoch, mIoU, nIoU, save_dir):
    plt.figure()
    plt.plot(num_epoch, mIoU, label='mIoU')
    plt.plot(num_epoch, nIoU, label='nIoU')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.savefig(save_dir + '/' + "fig_IoU.png")


def drawing_f1(num_epoch, f1, save_dir):
    plt.figure()
    plt.plot(num_epoch, f1, label='F1-score')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ylabel('F1-score')
    plt.xlabel('Epoch')
    plt.savefig(save_dir + '/' + "fig_F1-score.png")
