import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
from src import dataset, network, loss, log
from metrics import psnr, ssim
import math
from visdom import Visdom
import time


if __name__ == '__main__':
    # write log
    sys.stderr = log.Logger(sys.stderr)
    sys.stdout = log.Logger(sys.stdout)

    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameter
    EPOCH = 12
    LR = 0.0002
    BATCHSIZE = 16
    EM_MOM = 0.9

    # magnitude
    x_train_path = "./data/train/x"
    y_train_path = "./data/train/y"

    # model path
    save_path = "./model/PISNet"
    # save_path = "./model/IMN"

    # xdata_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize((), ())])
    # ydata_transform = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Normalize((), ())])

    train_data = dataset.readDataset(x_train_path, y_train_path)

    train_dataloader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=1)

    Net = network.PISNet().to(device)
    # Net = network.IMN().to(device)
    print(Net)

    loss_func = loss.LossCriterion()
    # loss_func = loss.IMNLoss()

    optimizer = torch.optim.Adam(Net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    print("---------------Start Training---------------")
    train_start = time.time()
    for epoch in range(EPOCH):
        epoch_start = time.time()
        print("### Epoch %d ###" % (epoch+1))
        print("Learning Rate of this epoch：%f" % (optimizer.param_groups[0]['lr']))
        for i, (src_data, trg_data) in enumerate(train_dataloader):
            src_data = src_data.to(device)
            trg_data = trg_data.to(device)

            out_data, mu = Net(src_data)
            # out_data = Net(src_data)

            loss, mse, low, spr = loss_func(out_data, trg_data, src_data)
            # loss = loss_func(out_data, trg_data)

            # Initialization of TFRM
            with torch.no_grad():
                mu = mu.mean(dim=0, keepdim=True)
                momentum = EM_MOM
                Net.tfrm.mu *= momentum
                Net.tfrm.mu += mu * (1 - momentum)

            optimizer.zero_grad()      # clear gradients for this training step
            loss.backward()            # back-propagation, compute gradients
            optimizer.step()           # apply gradients

            # use metrics
            out_numpy = out_data[0].squeeze(0).detach().cpu().numpy()
            trg_numpy = trg_data[0].squeeze(0).detach().cpu().numpy()
            SSIM = ssim.calculate_ssim(trg_numpy, out_numpy)
            PSNR = psnr.calculate_psnr(trg_numpy, out_numpy)

            # visualization
            if i % 4 == 0:
                print("epoch{0} batch{1} : Loss = {2}, MSE = {3}, LowRank = {4}, Sparse = {5}, SSIM = {6}, PSNR = {7}"
                      .format(epoch+1, i, loss, mse, low, spr, SSIM, PSNR))
                # print("epoch{0} batch{1} : Loss = {2}, SSIM = {3}, PSNR = {4}".format(epoch+1, i, loss, SSIM, PSNR))

        # time of every epoch
        epoch_end = time.time()
        print('Training time of Epoch ' + str(epoch+1) + ' is ' + str(epoch_end - epoch_start) + ' seconds.')

        # save model and update LR
        torch.save(Net, save_path + '/Model_epoch' + str(epoch+1) + '.pth')
        torch.save(Net.state_dict(), save_path + '/ModelWeight_epoch' + str(epoch + 1) + '.pth')

        # change learning rate
        scheduler.step()

    print("---------------Complete the Training---------------")
    train_end = time.time()
    total_time = train_end - train_start
    print('Total Time: {:.0f}min {:.0f}sec'.format(total_time // 60, total_time % 60))

    print("done!")

