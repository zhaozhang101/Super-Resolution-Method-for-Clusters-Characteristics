import argparse
import os
import model
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import torch
import time
import tools
import sys
since = time.time()
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set the super resolution scale
scale = 16

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=70, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--scale", type=int, default=scale, help="scale factor of SR")
parser.add_argument("--length_sample", type=int, default=17, help="num of snapshot in a sample")
parser.add_argument("--sourcepath", type=str, default=os.path.join('data\Sample'),
                    help="path of dataset")
# name of the model
parser.add_argument("--modelname", type=str, default="temp",
                    help="name of the model for current training")
if scale == 2:
    select_snapshot = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    select_snapshot_test = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16]
elif scale == 4:
    select_snapshot = [0, 4, 8, 12, 16]
    select_snapshot_test = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16]
elif scale == 8:
    select_snapshot = [0, 8, 16]
    select_snapshot_test = [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 16]
elif scale == 16:
    select_snapshot = [0, 16]
    select_snapshot_test = [0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 16]
else:
    raise KeyError
parser.add_argument("--select_snapshot", type=int, default=select_snapshot, help="length of select_snapshot")
parser.add_argument("--select_snapshot_test", type=int, default=select_snapshot_test, help="length of select_snapshot")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 设置训练集和测试集
training_dataset = tools.MyDataset('LOS_Dense', opt)
test_dataset = tools.MyDataset('MLW', opt)
data_loaders = torch.utils.data.DataLoader(training_dataset, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=0, drop_last=True)
testdata_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=0, drop_last=True)

model_CLST = model.CLST_1(opt).cuda()
model_CLST = tools.resume(model_CLST, os.path.join('model',f'model_{scale}.pth'))
loss = model.Myloss()
loss_std = model.Std()
losstra = model.MSEloss()
optimizer = torch.optim.Adam(model_CLST.parameters(), lr=opt.lr)
list = ['epoch', 'power_train', 'x_train', 'y_train', 'z_train', 'all_train', 'power_test', 'x_test', 'y_test', 'z_test',
            'all_test','std_power','std_x','std_y','std_z']
data = pd.DataFrame([list])
data.to_csv(os.path.join('training_result', f'{opt.modelname}_{opt.scale}.csv'), mode='w', header=None, index=False)
sys.stdout = tools.Record(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_result', f'{opt.modelname}_{opt.scale}.txt'))

for epoch in range(opt.epochs):
    loss_all_avg = 0
    loss_power_avg = 0
    loss_x_avg = 0
    loss_y_avg = 0
    loss_z_avg = 0
    # ----------
    #  Training
    # ----------
    model_CLST.train()
    for power_input, x_input, y_input, z_input, power_gt, x_gt, y_gt, z_gt, mask, key_num in data_loaders:

        power_in, x_in, y_in, z_in = model_CLST(key_num, power_input, x_input, y_input, z_input)

        loss_power = losstra(power_gt, power_in, mask)
        loss_x = losstra(x_gt, x_in, mask)
        loss_y = losstra(y_gt, y_in, mask)
        loss_z = losstra(z_gt, z_in, mask)

        # loss_power = loss(power_gt, power_input, mask)
        # loss_x = loss(x_gt, x_input, mask)
        # loss_y = loss(y_gt, y_input, mask)
        # loss_z = loss(z_gt, z_input, mask)

        loss_CLST = loss_power + loss_x + loss_y + loss_z
        optimizer.zero_grad()
        loss_CLST.backward()
        optimizer.step()

        loss_power_avg = loss_power_avg + loss_power / len(data_loaders)
        loss_x_avg = loss_x_avg + loss_x / len(data_loaders)
        loss_y_avg = loss_y_avg + loss_y / len(data_loaders)
        loss_z_avg = loss_z_avg + loss_z / len(data_loaders)
        loss_all_avg = loss_all_avg + loss_CLST / len(data_loaders)

        if epoch==0:
            break

    # ----------
    #  Testing
    # ----------
    model_CLST.eval()
    with torch.no_grad():
        test_power_avg = 0;test_power_avg_std=0
        test_x_avg = 0;test_x_avg_std = 0
        test_y_avg = 0;test_y_avg_std = 0
        test_z_avg = 0;test_z_avg_std = 0
        test_all_avg = 0
        for power_input, x_input, y_input, z_input, power_gt, x_gt, y_gt, z_gt, mask, key_num in testdata_loader:

            power_in, x_in, y_in, z_in = model_CLST(key_num, power_input, x_input, y_input, z_input)

            loss_power = loss(power_gt, power_in, mask)
            loss_x = loss(x_gt, x_in, mask)
            loss_y = loss(y_gt, y_in, mask)
            loss_z = loss(z_gt, z_in, mask)

            loss_power_std = loss_std(power_gt,power_in,mask)
            loss_x_std = loss_std(x_gt, x_in, mask)
            loss_y_std = loss_std(y_gt, y_in, mask)
            loss_z_std = loss_std(z_gt, z_in, mask)

            # loss_power = loss(power_gt, power_input, mask)
            # loss_x = loss(x_gt, x_input, mask)
            # loss_y = loss(y_gt, y_input, mask)
            # loss_z = loss(z_gt, z_input, mask)
            # loss_power_std = loss_std(power_gt,power_input,mask)
            # loss_x_std = loss_std(x_gt, x_input, mask)
            # loss_y_std = loss_std(y_gt, y_input, mask)
            # loss_z_std = loss_std(z_gt, z_input, mask)

            loss_CLST = loss_power + loss_x + loss_y + loss_z

            test_power_avg = test_power_avg + loss_power / len(testdata_loader)
            test_x_avg = test_x_avg + loss_x / len(testdata_loader)
            test_y_avg = test_y_avg + loss_y / len(testdata_loader)
            test_z_avg = test_z_avg + loss_z / len(testdata_loader)
            test_all_avg = test_all_avg + loss_CLST / len(testdata_loader)

            test_power_avg_std = test_power_avg_std + loss_power_std / len(testdata_loader)
            test_x_avg_std = test_x_avg_std + loss_x_std / len(testdata_loader)
            test_y_avg_std = test_y_avg_std + loss_y_std / len(testdata_loader)
            test_z_avg_std = test_z_avg_std + loss_z_std / len(testdata_loader)

    print('==============', epoch, '==============')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("training loss:",
          f"CLST_power:{loss_power_avg.item():.4f}, CLST_x:{loss_x_avg.item():.4f}, CLST_y:{loss_y_avg.item():.4f}, CLST_z:{loss_z_avg.item():.4f}, All:{loss_all_avg.item():.4f}")
    print("test loss:    ",
          f"CLST_power:{test_power_avg.item():.4f}, CLST_x:{test_x_avg.item():.4f}, CLST_y:{test_y_avg.item():.4f}, CLST_z:{test_z_avg.item():.4f}, All:{test_all_avg.item():.4f}")
    print("test loss:    ",
          f"CLST_power:{test_power_avg_std.item():.4f}, CLST_x:{test_x_avg_std.item():.4f}, CLST_y:{test_y_avg_std.item():.4f}, CLST_z:{test_z_avg_std.item():.4f}, All:{test_all_avg.item():.4f}")

    list = [epoch + 1];list.append(loss_power_avg.item())
    list.append(loss_x_avg.item());list.append(loss_y_avg.item());list.append(loss_z_avg.item());list.append(loss_all_avg.item())
    list.append(test_power_avg.item())
    list.append(test_x_avg.item())
    list.append(test_y_avg.item())
    list.append(test_z_avg.item())
    list.append(test_all_avg.item())
    list.append(test_power_avg_std.item())
    list.append(test_x_avg_std.item())
    list.append(test_y_avg_std.item())
    list.append(test_z_avg_std.item())

    data = pd.DataFrame([list])
    data.to_csv(os.path.join('training_result', f'{opt.modelname}_{opt.scale}.csv'), mode='a', header=None, index=False)

    if epoch == opt.epochs - 1:
        last_model_wts = model_CLST.state_dict()
        save_path_last = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_result',
                                          f'{opt.modelname}_{opt.scale}.pth')
        torch.save(last_model_wts, save_path_last)


