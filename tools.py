from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import torch
from pylab import *
import errno
import os

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def mkdir_if_missing(dir_path):  # 不需care
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Record(object):  # 记录输出结果到txt中的函数
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:  # 如果路径存在，
            mkdir_if_missing(os.path.dirname(path))
            self.file = open(path, 'w')  # 打开路径
        else:
            print('The path provided is wrong!')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def resume(model, resume_path):
    if not os.path.isfile(resume_path):
        print('                           ')
        print('can not load the model_wts!')
    else:
        model_wts = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(model_wts, strict=True)
        print('                          ')
        print('load model_wts successful!')
    return model


class MyDataset(Dataset):
    def __init__(self, path, opts):
        super(MyDataset, self).__init__()
        self.args = opts
        self.data_path = os.path.join(opts.sourcepath, path)
        self.files = os.listdir(self.data_path)
        self.length = len(self.files)

    def __len__(self):
        return self.length

    #  obj power x y z
    #  0   1:num_clst+1    num_clst+1:2*num_clst+1
    def __getitem__(self, index):
        data_read = pd.read_csv(os.path.join(self.data_path, self.files[index]), header=None)
        data = np.array(data_read.values.tolist())

        num_clst = int(data[0, 0])
        power_gt = FloatTensor(data[:, num_clst + 1:num_clst * 2 + 1])
        x_gt = FloatTensor(data[:, num_clst * 2 + 1:num_clst * 3 + 1])
        y_gt = FloatTensor(data[:, num_clst * 3 + 1:num_clst * 4 + 1])
        z_gt = FloatTensor(data[:, num_clst * 4 + 1:num_clst * 5 + 1])

        power_ds = FloatTensor(data[self.args.select_snapshot, num_clst + 1:num_clst * 2 + 1]).unsqueeze(0).unsqueeze(0)
        x_ds = FloatTensor(data[self.args.select_snapshot, num_clst * 2 + 1:num_clst * 3 + 1]).unsqueeze(0).unsqueeze(0)
        y_ds = FloatTensor(data[self.args.select_snapshot, num_clst * 3 + 1:num_clst * 4 + 1]).unsqueeze(0).unsqueeze(0)
        z_ds = FloatTensor(data[self.args.select_snapshot, num_clst * 4 + 1:num_clst * 5 + 1]).unsqueeze(0).unsqueeze(0)

        power_input = F.interpolate(power_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
        x_input = F.interpolate(x_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
        y_input = F.interpolate(y_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
        z_input = F.interpolate(z_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()

        for i in np.arange(0, 17, self.args.scale):
            power_input[i, :] = power_gt[i, :]
            x_input[i, :] = x_gt[i, :]
            y_input[i, :] = y_gt[i, :]
            z_input[i, :] = z_gt[i, :]

        mask = torch.ones(size=[17, num_clst]).cuda()
        mask[0:: self.args.scale, :] = 0.001
        mask[power_gt == 0] = 0.001
        # a = power_gt.cpu().numpy()
        # b = power_input.cpu().numpy()
        # c = x_gt.cpu().numpy()
        # d = x_input.cpu().numpy()
        # e = y_gt.cpu().numpy()
        # f = y_input.cpu().numpy()
        # g = z_gt.cpu().numpy()
        # h = z_input.cpu().numpy()
        # i = mask.cpu().numpy()
        # l = mask.cpu().numpy()

        return power_input, x_input, y_input, z_input, power_gt, x_gt, y_gt, z_gt, mask, num_clst
