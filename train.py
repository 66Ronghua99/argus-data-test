import torch
import torch.utils
import torch.utils.data
import tqdm
import numpy as np
from torch import nn
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

    def to(self, device):
        self.device = device
        super().to(device)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_step=16):
        self.data = data
        self.time_step = time_step

    def __len__(self):
        return len(self.data) - self.time_step

    def __getitem__(self, index):
        x = self.data[index:index + self.time_step]
        return torch.tensor(x, dtype=torch.float32)



def train():
    LR = 0.001
    criterion = nn.MSELoss()
    data = np.loadtxt(train_data, delimiter=',')
    device_Num = len(data[0])
    time_step = 16
    dataset = TimeSeriesDataset(data, time_step)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    if not check_point:
        model = AE(device_Num, time_step)
    else:
        model = torch.load(check_point)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for e in tqdm.tqdm(range(epoch)):
        for data in data_loader:
            input = data.to(device)
            output = model(input)
            optim.zero_grad()
            loss = criterion(output, input)
            loss.backward()
            optim.step()
        if (e + 1) % 10 == 0:
            print(f'Epoch [{e + 1}/{epoch}], Loss: {loss.item():.4f}')
        if (e + 1) % se == 0:
            torch.save(model, save_model+str(e+1))


epoch=35000
se=1000
check_point=None
save_model='home1_checkpoint/7day_epoch_'
batch_size=1024
train_data='Home1/train_data_7day.csv'


def main():
    global epoch, se, check_point, batch_size, train_data, save_model
    import argparse
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('-d', '--train_data', type=str, help='Training dataset, default: train_data_7day.csv')
    parser.add_argument('-m', '--model', type=str, help='Name of the model')
    parser.add_argument('-e', '--epoch', type=int, help='Name of the model')
    parser.add_argument('-sm', '--save_model', type=str, help='Name of the model')

    # 解析参数
    args = parser.parse_args()
    if args.train_data:
        train_data = args.train_data
    if args.model:
        check_point = args.model
    if args.epoch:
        epoch = args.epoch
    if args.save_model:
        save_model = args.save_model

    train()


if __name__ == "__main__":
    main()
