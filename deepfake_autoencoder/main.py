from preprocess import SourceDataset, TargetDataset
from model import SourceAE, TargetAE

def train(model, train_loader):
  with experiment.train():
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams["lr"])

    for epoch in tqdm(range(hyperparams["epochs"])):
        for batch in train_loader:
          img = batch["source_img"].to(device=device, dtype=torch.float)
          img_pred = model(img)

          optimizer.zero_grad()
          loss = loss_fn(img_pred, img)
          loss.backward()
          optimizer.step()

def inference(model, train_loader):
  with experiment.test():
    for batch in train_loader:
      img = batch["source_img"].to(device=device, dtype=torch.float)
      img_pred = model(img)

      img = img.cpu().detach().numpy() 
      img_pred = img_pred.cpu().detach().numpy()

      img = np.swapaxes(img, -1, 1)
      img_pred = np.swapaxes(img_pred, -1, 1)


      for i in range(hyperparams["bsz"]):
        img_ = img[i]
        img_pred_ = img_pred[i]
        print("original image")
        plt.imshow(img_)
        plt.show()

        print("reconstructed image")
        plt.imshow(img_pred_)
        plt.show()
      break 

def face_swap(source_loader, target_model):
  with experiment.test():
    for batch in source_loader:
      img = batch["source_img"].to(device=device, dtype=torch.float)
      img_pred = target_model(img)

      img = img.cpu().detach().numpy() 
      img_pred = img_pred.cpu().detach().numpy()

      img = np.swapaxes(img, -1, 1)
      img_pred = np.swapaxes(img_pred, -1, 1)


      for i in range(hyperparams["bsz"]):
        img_ = img[i]
        img_pred_ = img_pred[i]
        print("original image")
        plt.imshow(img_)
        plt.show()

        print("swapped image")
        plt.imshow(img_pred_)
        plt.show()
      break 

if __name__ == "__main__":
    experiment = Experiment(api_key="CPjn0JE1SVdtVaV3f8k2zwYMC",
                        project_name="deepfake", workspace="minjeancho")
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE NAME: ", torch.cuda.get_device_name(0))

    hyperparams = {
        "lr": 0.001,
        "epochs": 100,
        "bsz": 16
    }

    source_model = SourceAE().to(device)
    target_model = TargetAE().to(device)

    print("Extracting source data...")
    source_dataset = SourceDataset("/data/trump")
    print("Done!")
    source_train_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size = hyperparams["bsz"], shuffle=True)
    # === train source AE with source images 
    print("Training source autoencoder...")
    train(source_model, source_train_loader)

    print("Extracting target data...")
    target_dataset = TargetDataset("/data/cage")
    print("Done!")
    target_train_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size = hyperparams["bsz"], shuffle=True)
    # == target encoder shares weight with source encoder 
    target_model.encoder.load_state_dict(source_model.encoder.state_dict())
    # === train target AE with target images 
    print("Training target autoencoder...")
    train(target_model, target_train_loader)


    #########################################
    #########################################
    print("Reconstructing source dataset...")
    inference(source_model, source_train_loader)
    print("Reconstructing target dataset...")
    inference(target_model, target_train_loader)
    print("Face swapping...")
    face_swap(source_train_loader, target_model)