import argparse

import torch
from torch.utils.data import DataLoader

from model import ModelClass
from utils import RecommendationDataset

from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--testset', default='./test', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='train loader batch size')

    args = parser.parse_args()

    
    # load dataset in train folder
    train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  
    n_users, n_items, n_ratings = train_data.get_datasize()
    
    # instantiate model
    model = ModelClass(n_users, n_items, rank = 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    
    for epoch in range(20):
        cost = 0
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            ratings_pred = model(users, items)
            loss = criterion(ratings_pred, ratings)
            loss.backward()
            optimizer.step()
            cost += loss.item() * len(ratings)
            
        cost /= n_ratings
        
        print(f"Epoch: {epoch}")
        print("train cost: {:.6f}" .format(cost))
        
        cost_test = 0


    torch.save(model.state_dict(), args.save_model)
