from preprocess import DeepfakeDataset
from model import SourceAE, TargetAE


def train(model, train_loader):
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams["lr"])

    for epoch in range (hyperparams["epochs"]):
        for batch in train_loader:
        x = batch.to(device)
        x_pred = model(x)

        optimizer.zero_grad()
        loss = loss_fn(x_pred, x)
        loss.backward()
        optimizer.step()

#TODO: test loop 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("DEVICE NAME: ", torch.cuda.get_device_name(0))

    #TODO: finetune hyperparams 
    hyperparams = {
        "lr": 0.001,
        "epochs": 100,
    }

    source_model = SourceAE()
    target_model = TargetAE()


    source_dataset = DeepfakeDataset()
    source_train_loader = torch.utils.data.DataLoader(dataset=source_dataset, shuffle=True)
    # === train source AE with source images 
    train(source_model, source_train_loader)

    target_dataset = DeepfakeDataset()
    target_train_loader = torch.utils.data.DataLoader(dataset=target_dataset, shuffle=True)
    # == target encoder shares weight with source encoder 
    target_model.load_state_dict(source_model.encoder.state_dict())
    # === train target AE with target images 
    train(target_model, target_train_loader)
            