import scipy.io as sio
import csv
import pandas as pd
import glob
import torch.nn.functional as F
import torch
from pylab import *
import tools
import scipy.io
import os
from model import CLST_1

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def removekeys(tmp):  # 检测是否有x00字符，如果有，去除掉
    old_list = list(tmp.keys())
    tip = False
    old_list = list(tmp.keys())
    for element in old_list:
        if '\x00' in element:
            tip = True
    if tip == True:
        new_list = [x.replace('\x00', '') for x in old_list]
        for element in range(len(new_list)):
            tmp[new_list[element]] = tmp[old_list[element]]
            del tmp[old_list[element]]
    else:
        tmp = tmp
    return tmp


def avgDividCluster(keyMatrix, T0,Tx):
    index = np.zeros((keyMatrix.shape[0]))
    index[keyMatrix[:, 2] < T0 + 1e-6] = 1
    keyMatrix = keyMatrix[index == 1, :]
    numCluster, former, count = 0, 0, 0

    tmp = np.argsort(keyMatrix[:, 17])
    keyMatrix = keyMatrix[tmp, :]
    objId = keyMatrix[0, 17]
    keycell = []
    for idx0 in range(keyMatrix.shape[0] - 1):
        if objId != keyMatrix[idx0, 17]:
            numCluster += 1
            objId = keyMatrix[idx0, 17]
            later = idx0
            keycell.append(keyMatrix[former:later, :])
            former = idx0
            count += 1
    later = keyMatrix.shape[0] - 1
    keycell.append(keyMatrix[former:later, :])
    K_cell = keycell
    # 等分分簇
    '''
    for idx1 in range(len(keycell)):
        delayKeycell = keycell[idx1][:, 2]
        idx_tmp = np.zeros((delayKeycell.shape[0]))

        delayMax = max(keycell[idx1][:, 2])
        delayMin = min(keycell[idx1][:, 2])
        delaySpan = delayMax - delayMin
        span = delaySpan * 1e9
        if span > 400:  # 等分四份
            one_four = delayMin + delaySpan / 4
            two_four = delayMin + delaySpan / 2
            three_four = delayMin + delaySpan * 3 / 4
            idx_tmp[delayKeycell < one_four] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[(one_four <= delayKeycell) & (delayKeycell < two_four)] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[(two_four <= delayKeycell) & (delayKeycell < three_four)] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[three_four <= delayKeycell] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

        elif span > 300:
            one_three = delayMin + delaySpan / 3
            two_three = delayMin + delaySpan * 2 / 3
            idx_tmp[delayKeycell < one_three] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[(one_three <= delayKeycell) & (delayKeycell < two_three)] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[two_three <= delayKeycell] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

        elif span > 200:
            one_two = delayMin + delaySpan / 2
            idx_tmp[delayKeycell < one_two] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

            idx_tmp[one_two <= delayKeycell] = 1
            if sum(idx_tmp) != 0:
                K_cell.append(keycell[idx1][idx_tmp == 1, :])
                idx_tmp = np.zeros((delayKeycell.shape[0]))

        else:
            K_cell.append(keycell[idx1])
            idx_tmp = np.zeros((delayKeycell.shape[0]))
    '''
    objectID, cluCenterX, cluCenterY, cluCenterZ, cluCenterP = [], [], [], [], []
    for idx2 in range(len(K_cell)):
        powerSum, clusterPowerComplex, X, Y, Z = 0, 0, 0, 0, 0
        for idx3 in range(K_cell[idx2].shape[0]):
            power = K_cell[idx2][idx3, 4] ** 2 + K_cell[idx2][idx3, 5] ** 2
            powerSum += power
            clusterPowerComplex += complex(K_cell[idx2][idx3, 4], K_cell[idx2][idx3, 5])
            X += power * K_cell[idx2][idx3, 12]
            Y += power * K_cell[idx2][idx3, 13]
            Z += power * K_cell[idx2][idx3, 14]
        objectID.append(K_cell[idx2][0, 17])
        cluCenterX.append(X / powerSum)
        cluCenterY.append(Y / powerSum)
        cluCenterZ.append(Z / powerSum)
        cluCenterP.append(np.abs(clusterPowerComplex) ** 2)
    cluCenterX, cluCenterY, cluCenterZ, objectID = np.asarray(cluCenterX), np.asarray(cluCenterY), np.asarray(
        cluCenterZ), np.asarray(objectID)
    if K_cell[0][0,0]==0:
        cluCenterX[0] = Tx[0]+1e-4
        cluCenterY[0] = Tx[1]+1e-4
        cluCenterZ[0] = Tx[2]+1e-4
    # 删去功率较小的簇 阈值90dB
    cluCenterP = 10 * np.log10(cluCenterP)
    cluCenterP_max = max(cluCenterP)
    idx4 = np.zeros((len(K_cell)))
    idx4[cluCenterP > cluCenterP_max - 90] = 1
    cluCenterP = cluCenterP[idx4 == 1]
    cluCenterX = cluCenterX[idx4 == 1]
    cluCenterY = cluCenterY[idx4 == 1]
    cluCenterZ = cluCenterZ[idx4 == 1]
    objectID = objectID[idx4 == 1]
    return cluCenterP, cluCenterX, cluCenterY, cluCenterZ, objectID


def extractCluster(sourcepath,file):
    files = glob.glob(os.path.join(sourcepath, '*_snapshot*.mat'))
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(files)):
            i += 1
            snapshot = sio.loadmat(sourcepath + '/result_snapshot_' + str(i) + '.mat')
            snapshot = removekeys(snapshot)
            vector = snapshot['TxRx'][0, :] - snapshot['TxRx'][1, :]
            T0 = np.sqrt(sum(vector ** 2)) / 3e8
            if snapshot['RaysProperties'].shape[
                1] < 15:  # 如果这个snapshot只有一条直射径
                continue

            Cst_cp, Cst_cx, Cst_cy, Cst_cz, obj_id = avgDividCluster(snapshot['RaysProperties'], T0,snapshot['TxRx'][0, :])
            Cluster_center_X = Cst_cx - snapshot['TxRx'][0, 0]
            Cluster_center_Y = Cst_cy - snapshot['TxRx'][0, 1]
            Cluster_center_Z = Cst_cz - snapshot['TxRx'][0, 2]
            Num_cluster = np.array([len(Cst_cp)])

            tmp = np.concatenate((Num_cluster, obj_id, Cst_cp, Cluster_center_X, Cluster_center_Y, Cluster_center_Z),
                                 axis=0)
            a = list(tmp)
            writer.writerow(list(tmp))
    print('Extract Done!')


def matchCluster(file,samplepath):
    data = pd.read_csv(file,header=None, usecols=[0]).to_numpy().squeeze()
    max = np.max(data)

    colum_count = int(max * 5 + 1)
    column_names = [i for i in range(0, colum_count)]
    data_all = pd.read_csv(file,header=None,names=column_names)
    data_all = data_all.to_numpy()
    for idx in range(data_all.shape[0] - 16):
        with open(samplepath +'/'+ f'{idx}.csv', 'w', encoding='UTF8', newline='') as f:
            print("Snapshot:",idx)
            # 第一阶段获取基本信息 cluster_num第一列簇数量信息
            # data_batch一个样本内的数据    key为最小簇数量的snapshot的数据 compro_num 本样本中折中的簇数量
            cluster_num = data_all[idx:idx + 17, 0]
            max_num, min_num = int(np.max(cluster_num)), int(np.min(cluster_num))
            data_batch = data_all[idx:idx + 17, 0:5 * max_num + 1]
            key_cluster = np.where(cluster_num == min_num)[0][0]
            # key_cluster = key_cluster(1)
            key_obj = data_batch[key_cluster, 1:min_num + 1]
            key_power = data_batch[key_cluster, min_num + 1:2 * min_num + 1]
            key_x = data_batch[key_cluster, 2 * min_num + 1:3 * min_num + 1]
            key_y = data_batch[key_cluster, 3 * min_num + 1:4 * min_num + 1]
            key_z = data_batch[key_cluster, 4 * min_num + 1:5 * min_num + 1]
            compro_num = np.floor(min_num / 5) * 5
            compro_num = 1 if compro_num == 0 else compro_num
            compro_num = 45 if compro_num > 45 else compro_num
            compro_num = int(compro_num)

            # 第二阶段 取功率较大的compronum个簇
            idx2 = np.argsort(-key_power).squeeze()
            idx3 = np.zeros((min_num))
            idx3[idx2[0:compro_num]] = 1
            key_obj = key_obj[idx3 == 1]
            key_obj[key_obj == 0] = 1
            key_power = key_power[idx3 == 1]
            key_x = key_x[idx3 == 1]
            key_y = key_y[idx3 == 1]
            key_z = key_z[idx3 == 1]

            # 第三阶段 写入文件
            for idx4 in range(17):
                target_num = int(cluster_num[idx4])
                target_obj = data_batch[idx4, 1:target_num + 1]
                target_obj[target_obj == 0] = -1
                targetPower = data_batch[idx4, target_num + 1:2 * target_num + 1]
                targetX = data_batch[idx4, 2 * target_num + 1:3 * target_num + 1]
                targetY = data_batch[idx4, 3 * target_num + 1:4 * target_num + 1]
                targetZ = data_batch[idx4, 4 * target_num + 1:5 * target_num + 1]
                idx5 = np.isin(target_obj, key_obj).astype(int)

                new_obj, new_power, new_x, new_y, new_z = np.zeros((compro_num)), np.zeros((compro_num)), np.zeros(
                    (compro_num)), np.zeros((compro_num)), np.zeros((compro_num))
                for idx6 in range(compro_num):
                    if np.isin(key_obj[idx6], target_obj):
                        info_idx = np.where(key_obj[idx6] == target_obj)[0]
                        x_real = np.array(abs(targetX[info_idx] - key_x[idx6]) < 10, dtype=int)
                        y_real = np.array(abs(targetY[info_idx] - key_y[idx6]) < 10, dtype=int)
                        z_real = np.array(abs(targetZ[info_idx] - key_z[idx6]) < 10, dtype=int)
                        arr = z_real   # x_real * y_real * z_real
                        info_idx = [info_idx[i] for i in range(len(info_idx)) if arr[i] == 1]
                        if np.sum(info_idx) > 1:
                            info_idx = info_idx[0]
                        if np.sum(arr) > 0:
                            new_obj[idx6] = target_obj[info_idx]
                            new_power[idx6] = targetPower[info_idx]
                            new_x[idx6] = targetX[info_idx]
                            new_y[idx6] = targetY[info_idx]
                            new_z[idx6] = targetZ[info_idx]
                tmp = np.concatenate((np.array([compro_num]), new_obj, new_power, new_x, new_y, new_z), axis=0)
                writer = csv.writer(f)
                writer.writerow(tmp)

    print("Match Done!")


def predict(samplepath,savepath,scale,model_name)->None:
    if scale == 2:
        select_snapshot = [0, 2, 4, 6, 8, 10, 12, 14, 16]  # 相当于低分辨率下 的真值
    elif scale == 4:
        select_snapshot = [0, 4, 8, 12, 16]
    elif scale == 8:
        select_snapshot = [0, 8, 16]
    elif scale == 16:
        select_snapshot = [0, 16]
    else:
        raise KeyError
    minind = 0  # snapshot起始
    maxind = 99  # snapshot截止
    # 重载模型参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLST_1(opt = None).to(device)
    model = tools.resume(model, model_name)

    index_min = minind
    index_max = maxind
    # length = index_max - index_min + 1
    length = 17
    # length = len(range(index_min, index_max, 17))*17  相邻17个snapshot为一个样本csv文件 ，故隔17个样本选择一个样本
    # 在mat代码生成簇时 设置簇数量最大是45
    power_pre = np.zeros(shape=[length, 45])
    power_grdth = np.zeros(shape=[length, 45])
    power_bs = np.zeros(shape=[length, 45])
    x_pre = np.zeros(shape=[length, 45])
    x_grdth = np.zeros(shape=[length, 45])
    x_bs = np.zeros(shape=[length, 45])
    y_pre = np.zeros(shape=[length, 45])
    y_grdth = np.zeros(shape=[length, 45])
    y_bs = np.zeros(shape=[length, 45])
    z_pre = np.zeros(shape=[length, 45])
    z_grdth = np.zeros(shape=[length, 45])
    z_bs = np.zeros(shape=[length, 45])

    for root, dirs, files in os.walk(samplepath):
        for file in files:
            if file.endswith(".csv"):
                data_read = pd.read_csv(os.path.join(samplepath, f"{index_max - 16}.csv"),header=None)

                data = np.array(data_read.values.tolist())
                num_clst = int(data[0, 0])

                power_gt = FloatTensor(data[:, num_clst + 1:num_clst * 2 + 1])
                x_gt = FloatTensor(data[:, num_clst * 2 + 1:num_clst * 3 + 1])
                y_gt = FloatTensor(data[:, num_clst * 3 + 1:num_clst * 4 + 1])
                z_gt = FloatTensor(data[:, num_clst * 4 + 1:num_clst * 5 + 1])
                obj_gt = FloatTensor(data[:, 1:num_clst + 1])

                power_ds = FloatTensor(data[select_snapshot, num_clst + 1:num_clst * 2 + 1]).unsqueeze(0).unsqueeze(0)
                x_ds = FloatTensor(data[select_snapshot, num_clst * 2 + 1:num_clst * 3 + 1]).unsqueeze(0).unsqueeze(0)
                y_ds = FloatTensor(data[select_snapshot, num_clst * 3 + 1:num_clst * 4 + 1]).unsqueeze(0).unsqueeze(0)
                z_ds = FloatTensor(data[select_snapshot, num_clst * 4 + 1:num_clst * 5 + 1]).unsqueeze(0).unsqueeze(0)

                power_input = F.interpolate(power_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
                x_input = F.interpolate(x_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
                y_input = F.interpolate(y_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()
                z_input = F.interpolate(z_ds, size=[17, num_clst], mode='bilinear', align_corners=True).squeeze()

                for i in np.arange(0, 17, scale):
                    power_input[i, :] = power_gt[i, :]
                    x_input[i, :] = x_gt[i, :]
                    y_input[i, :] = y_gt[i, :]
                    z_input[i, :] = z_gt[i, :]

                mask = torch.ones(size=[17, num_clst])
                mask[0:: scale, :] = 0.001
                mask[power_gt == 0] = 0.001

                power_input = power_input.unsqueeze(0)
                x_input = x_gt.unsqueeze(0)
                y_input = y_gt.unsqueeze(0)
                z_input = z_gt.unsqueeze(0)

                power_in, x_in, y_in, z_in = model(num_clst, power_input, x_input, y_input, z_input)

                power_in = power_in.squeeze()
                x_in = x_in.squeeze()
                y_in = y_in.squeeze()
                z_in = z_in.squeeze()

                for i in np.arange(0, 17, scale):
                    power_in[i, :] = power_gt[i, :]
                    x_in[i, :] = x_gt[i, :]
                    y_in[i, :] = y_gt[i, :]
                    z_in[i, :] = z_gt[i, :]

                power_in = power_in.detach().cpu().numpy()
                x_in = x_in.detach().cpu().numpy()
                y_in = y_in.detach().cpu().numpy()
                z_in = z_in.detach().cpu().numpy()

                power_input = power_input.squeeze().cpu().numpy()
                x_input = x_input.squeeze().cpu().numpy()
                y_input = y_input.squeeze().cpu().numpy()
                z_input = z_input.squeeze().cpu().numpy()

                keys = power_in.shape[1]  # 簇数量

                index_cou = 0
                power_bs[index_cou:index_cou + 17, 0:keys] = power_input
                x_bs[index_cou:index_cou + 17, 0:keys] = x_input
                y_bs[index_cou:index_cou + 17, 0:keys] = y_input
                z_bs[index_cou:index_cou + 17, 0:keys] = z_input

                power_pre[index_cou:index_cou + 17, 0:keys] = power_in
                x_pre[index_cou:index_cou + 17, 0:keys] = x_in
                y_pre[index_cou:index_cou + 17, 0:keys] = y_in
                z_pre[index_cou:index_cou + 17, 0:keys] = z_in

                power_grdth[index_cou:index_cou + 17, 0:keys] = power_gt.cpu().numpy()
                x_grdth[index_cou:index_cou + 17, 0:keys] = x_gt.cpu().numpy()
                y_grdth[index_cou:index_cou + 17, 0:keys] = y_gt.cpu().numpy()
                z_grdth[index_cou:index_cou + 17, 0:keys] = z_gt.cpu().numpy()

            # 储存功率和交点的 真值与预测值
            scipy.io.savemat(f'prediction/{savepath}_{scale}_{file[:-4]}.mat', mdict={
                'power_pre': power_pre, 'x_pre': x_pre, 'y_pre': y_pre, 'z_pre': z_pre,
                'power_grdth': power_grdth, 'x_grdth': x_grdth, 'y_grdth': y_grdth, 'z_grdth': z_grdth,
                'power_bs': power_bs, "x_bs": x_bs, "y_bs": y_bs, "z_bs": z_bs})

