import torch
from torch import nn
import torch.fft

class Myloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, output, groundtruth, mask): # [0,1,2]
        yhat = torch.mul(output, mask)
        gtruth = torch.mul(groundtruth, mask)
        predot = torch.sum(mask, dim=[1, 2])
        ans = torch.sum(self.loss(yhat, gtruth), dim=[1, 2]) / predot
        ans = torch.mean(ans, dim=0)
        return ans

class MSEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, output, groundtruth, mask): # [b,s,f]
        yhat = torch.mul(output, mask)
        gtruth = torch.mul(groundtruth, mask)
        predot = torch.sum(mask, dim=[1, 2])
        ans = torch.sum(self.loss(yhat, gtruth), dim=[1, 2]) / predot
        ans = torch.mean(ans, dim=0)
        return ans

class Std(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, output, groundtruth, mask): # [b,s,f]
        """
        函数功能：计算真实值和预测值之间的标准差（STDE）
        :param y_hat: 预测值
        :param g_truth: 真实值
        :param mask: 蒙板
        :return: std标准差
        """
        yhat = torch.mul(output, mask)
        gtruth = torch.mul(groundtruth, mask)
        predot = torch.sum(mask, dim=[1, 2]) #b 1 1
        loss = yhat - gtruth
        mean = torch.div(torch.sum(loss, dim=[1, 2], keepdim=True), predot)
        para1 = torch.mul((loss-mean)**2, mask)
        para2 = torch.sqrt(torch.sum(para1,dim =[1,2],keepdim=True)/predot)
        para2 = torch.mean(para2,dim=0)
        return para2

class CLST(nn.Module):
    def __init__(self, opt):
        super(CLST, self).__init__()
        self.model5 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model6 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model7 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model8 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )

    def forward(self, key_num, power_input, x_input, y_input, z_input):
        # shape(power_input, x_input, y_input, z_input) :
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]

        power_input = power_input.transpose(1, 2)
        x_input = x_input.transpose(1, 2)
        y_input = y_input.transpose(1, 2)
        z_input = z_input.transpose(1, 2)

        power_in = self.model5(power_input) + power_input
        power_in = self.model5(power_in) + power_input
        power_in = self.model5(power_in) + power_input

        x_in = self.model6(x_input) + x_input
        x_in = self.model6(x_in) + x_input
        x_in = self.model6(x_in) + x_input

        y_in = self.model7(y_input) + y_input
        y_in = self.model7(y_in) + y_input
        y_in = self.model7(y_in) + y_input

        z_in = self.model8(z_input) + z_input
        z_in = self.model8(z_in) + z_input
        z_in = self.model8(z_in) + z_input

        power_in = power_in.transpose(1, 2)
        x_in = x_in.transpose(1, 2)
        y_in = y_in.transpose(1, 2)
        z_in = z_in.transpose(1, 2)

        return power_in, x_in, y_in, z_in

class CLST_1(nn.Module):
    def __init__(self,opt):
        super(CLST_1, self).__init__()
        self.model5 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model6 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model7 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )
        self.model8 = nn.Sequential(
            nn.Linear(17, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 17),
            nn.LeakyReLU(0.01),
        )

    def forward(self, key_num, power_input, x_input, y_input, z_input):
        # shape(power_input, x_input, y_input, z_input) :
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]
        # [batch_size, 样本中snapshot数量（17） , 本样本中对齐的簇的数量（例如5/10/15/20/25）]

        power_input = power_input.transpose(1, 2)
        x_input = x_input.transpose(1, 2)
        y_input = y_input.transpose(1, 2)
        z_input = z_input.transpose(1, 2)

        power_in = self.model5(power_input) + power_input
        power_in = self.model5(power_in) + power_input
        power_in = self.model5(power_in)

        x_in = self.model6(x_input) + x_input
        x_in = self.model6(x_in) + x_input
        x_in = self.model6(x_in)

        y_in = self.model7(y_input) + y_input
        y_in = self.model7(y_in) + y_input
        y_in = self.model7(y_in)

        z_in = self.model8(z_input) + z_input
        z_in = self.model8(z_in) + z_input
        z_in = self.model8(z_in)

        power_in = power_in.transpose(1, 2)
        x_in = x_in.transpose(1, 2)
        y_in = y_in.transpose(1, 2)
        z_in = z_in.transpose(1, 2)

        return power_in, x_in, y_in, z_in