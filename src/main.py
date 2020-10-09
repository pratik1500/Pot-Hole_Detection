import load_data
import train
import model
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


train_data, test_data = load_data.data__loader()
net = model.Net()

# checking if the gpu is available or not
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("cuda")
else:
    device = "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# this for the commit
data = transform(train_data)

trainloader = torch.utils.data.DataLoader(
    data, batch_size=64, shuffle=True)
load_data.plot(trainloader)

print("all runned well")

net.to(device=device)
trained_model = train.train_model(net, train_data)


#print("model is trained")
#train.test_model(trained_model, test_data)
