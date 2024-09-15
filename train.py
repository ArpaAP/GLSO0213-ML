import torch
import torch.nn as nn
import torch.optim as optim
import time

def train():
    start_time = time.time()

    with open('inputs.txt', 'r') as f_in:
        lines = f_in.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split()
        lines[i] = list(map(int, lines[i]))

    inputs = []
    outputs = []

    for lineVal in lines:
        line = lineVal[:-1]
        inputs.append(line)

        line = [0.1 for _ in range(5)]
        line[lineVal[-1]] = 0.9
        outputs.append(line)

    device = 'cpu'

    X = torch.FloatTensor(inputs).to(device)
    Y = torch.FloatTensor(outputs).to(device)

    try:
        model = torch.load('trained.wgt')
    except:
        n_hidden = 128 * 128
        model = nn.Sequential(
            nn.Linear(128 * 128, n_hidden, bias=True),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_hidden - 6500, bias=True),
            nn.Sigmoid(),
            nn.Linear(n_hidden - 6500, n_hidden-13000, bias=True),
            nn.Sigmoid(),
            nn.Linear(n_hidden-13000, 5, bias=True),
            nn.Sigmoid()
        ).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        optimizer.zero_grad()

        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, 100, cost.item()))

    end_time = time.time()

    print('Time: ', end_time - start_time)
    
    torch.save(model, 'trained.wgt')
    return

    
if __name__ == '__main__':
    train()