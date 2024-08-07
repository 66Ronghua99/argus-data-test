import torch
from torch import  nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AE(nn.Module):
    def __init__(self, device_num, length) -> None:
        super().__init__()
        self.device_num = device_num
        self.length = length
        self.encoder_gru1 = nn.GRU(input_size=device_num, hidden_size=256, batch_first=True)
        self.encoder_drop = nn.Dropout(p=0.3)
        self.encoder_gru2 = nn.GRU(input_size=256, hidden_size=64, batch_first=True)
        self.decoder_gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.decoder_drop = nn.Dropout(p=0.3)
        self.decoder_gru2 = nn.GRU(input_size=64, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, device_num)

    def forward(self, x):
        batch_num = x.size()[0]
        output, _ = self.encoder_gru1(x)
        output = self.encoder_drop(output)
        _, lat_embed = self.encoder_gru2(output)

        lat_embed = lat_embed.reshape((batch_num, 1, 64))
        lat_embed = lat_embed.repeat((1, self.length, 1))

        output, _ = self.decoder_gru1(lat_embed)
        output = self.decoder_drop(output)
        output, _ = self.decoder_gru2(output)
        output = self.fc(output)
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_step=1):
        self.data = data
        self.time_step = time_step

    def __len__(self):
        return len(self.data) - self.time_step

    def __getitem__(self, index):
        x = self.data[index:index + self.time_step]
        return torch.tensor(x, dtype=torch.float32)
        

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('-d', '--train_data', type=str, help='Training dataset, default: train_data_7day.csv')
    parser.add_argument('-td', '--test_data', type=str, help='Testing dataset')
    parser.add_argument('-m', '--model', type=str, help='Name of the model')

    args = parser.parse_args()
    
    if args.model:
        model = torch.load(args.model)
    else:
        model = torch.load("home1_checkpoint/3day_epoch_20000")
    if args.train_data:
        train_data = np.loadtxt(args.train_data, delimiter=',')
    else:
        train_data = np.loadtxt('Home1/train_data_3day.csv', delimiter=',')
    if args.test_data:
        test_data = np.loadtxt(args.test_data, delimiter=',')
    else:
        test_data = np.loadtxt('Home1/test-2021-07-03.csv', delimiter=',')
    criterion = torch.nn.MSELoss()
    device_Num = len(train_data[0])
    time_step = 16
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_dataset = TimeSeriesDataset(train_data, time_step)
    train_data_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
    test_dataset = TimeSeriesDataset(test_data, time_step)
    test_data_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    test_dataset2 = TimeSeriesDataset(test_data, time_step)
    test_data_loader2 = DataLoader(test_dataset, batch_size=1, pin_memory=True)
    
    model.to(device)
    model.eval()
    # train_losses = []
    # print("Calculating threshold")
    # with torch.no_grad():
    #     for data in train_data_loader:
    #         data = data.to(device)
    #         output = model(data)
    #         loss = criterion(data, output)
    #         train_losses.append(loss)
    
    # train_losses = torch.Tensor(train_losses)
    # min_loss = torch.min(train_losses).item()
    # max_loss = torch.max(train_losses).item()

    # threshold = max_loss + 0.2*(max_loss-min_loss)
    threshold = 0.03933517899204162
    print("Threashold:", threshold)

    test_losses = []
    with torch.no_grad():
        for data in test_data_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            test_losses.append(loss.item())

    import  matplotlib.pyplot as plt

    losses = np.array(test_losses)
    time = np.array(list(range(len(losses))))
    plt.figure(figsize=(10, 6))
    plt.scatter(time, losses, color='blue', label='Loss Values')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

    # 标记位于threshold之上的点
    above_threshold = losses > threshold
    # plt.scatter(time[4001:4201], losses[4001:4201], color='green', label='Anomaly')
    plt.scatter(time[above_threshold], losses[above_threshold], color='red', label='Above Threshold')


    plt.xlabel('Time')
    plt.ylabel('Loss Value')
    plt.title('Loss Values Over Time with Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()



main()