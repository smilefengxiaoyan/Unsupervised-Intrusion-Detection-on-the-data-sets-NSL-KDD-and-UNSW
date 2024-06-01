import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class AE(nn.Module):

    def __init__(self, M_type, train_features, init_lr, active_f , input_size, hidden_size, output_size, train_dataset,num_train,train_size):
        super(AE, self).__init__()
        self.M_type = M_type
        self.train_features = train_features
        self.train_dataset = train_dataset
        self.init_lr = init_lr
        self.active_f = nn.ELU
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_train = num_train  # 训练的总轮数，表示整个数据集将被遍历多少次。
        self.train_size = train_size
        self.features_rank = []


        # Encoder
        # Ensure activation functions are used as layers within nn.Sequential
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            self.active_f(),  # Instantiate ELU as a module
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.active_f() # Again, instantiate ELU
        )
        self.latent = nn.Linear(hidden_size[1], hidden_size[2])
        self.decoder = nn.Sequential(

            nn.ELU(),  # Proper instantiation
            nn.Linear(hidden_size[2], output_size),
            nn.Sigmoid()  # Sigmoid for output activation
        )

        if M_type == "AEDropout":
            self.dropout = nn.Dropout(0.8)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        latent = self.latent(x)
        x = self.decoder(latent)

        return x, latent

def training(M_type, train_features, init_lr, active_f, input_size, hidden_size, output_size, train_dataset,num_train,train_size):

        batch_size = train_size
        torch.manual_seed(0)
        device = torch.device("cpu")

        model = AE(M_type, train_features, init_lr, active_f, input_size, hidden_size, output_size, train_dataset,num_train,train_size)

        optimizer = optim.Adam(model.parameters(), lr=init_lr)

        num_epoch = num_train

        x = []
        y = []
        rank = []



        loss_fun = nn.MSELoss()  # MSE

        train_dataloader = torch.utils.data.DataLoader(train_features, batch_size=batch_size, shuffle=True)

        for epoch in range(1, num_epoch):

            model.train()

            total_loss = 0

            for sample in train_dataloader:

                X = sample.to(device)

                optimizer.zero_grad()

                reconstructed, latent = model(X.float())
                loss = loss_fun(reconstructed, X.float())



                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print("epoch {} loss {}".format(epoch, total_loss / len(train_dataloader)))
            init_lr = init_lr * (0.7 ** (epoch // 25))

            x.append(epoch)
            y.append(total_loss / len(train_dataloader))
        # Save the model state
        model_path = './trained_model.pth'
        torch.save(model.state_dict(), model_path)
        # To plot variable importances
        for name, param in model.named_parameters():
            if param.requires_grad:
                if list(param.shape) == [43, 38] or list(param.shape) == [41, 35]:
                    rank = param
                    rank= rank.tolist()

        plt.rcParams['figure.figsize'] = (20, 20)
        plt.plot(x, y)
        plt.xlabel("Epochs")
        plt.ylabel("Autoencoder Training loss")
        plt.show()

        features_rank = feature_importance(rank)
        bf = select_features(train_features, train_dataset, features_rank)

        return bf

def feature_importance(rank):

        features_rank = []

        for row in rank:
            total_weight = 0
            for a in row:
                total_weight += a
            features_rank.append(total_weight)

        return features_rank

def select_features(training_data, train_dataset, features_rank):
        features = list(range(0, training_data.shape[1]))
        feature_names = list(train_dataset.columns)

        plt.close()
        plt.rcParams['figure.figsize'] = (20, 20)
        x_vals = np.arange(len(features))
        #print(x_vals.shape)
        #print(len(features_rank))
        plt.bar(x_vals, features_rank, align='center', alpha=1)
        plt.xticks(x_vals, features)
        plt.xlabel("Feature Indices")
        plt.ylabel("Feature Importance Values")
        plt.show()

        selct_features = sorted(range(len(features_rank)), key=lambda i: features_rank[i], reverse=True)[:20]
        print("\nTop 20 selected features:")
        print(selct_features)
        for i in selct_features:
            print(feature_names[i])

        print("\nVariance", np.var(features_rank))

        return selct_features

# Function to calculate reconstruction errors
def calculate_reconstruction_errors(data_loader, model):
    model.eval()
    reconstruction_errors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data in data_loader:
        # Check if data is a tuple which often happens with DataLoader objects where data and labels are returned
        if isinstance(data, tuple) and len(data) > 0:
            data = data[0]  # Assuming data is the first element of the tuple

        # Convert Pandas DataFrame to tensor if that's the data type
        if isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values, dtype=torch.float32)

        # Ensure data is already a tensor (which it should be if properly prepared)
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a tensor but got type: {}".format(type(data)))

        data = data.to(device)

        with torch.no_grad():
            reconstructed, laten = model(data)
            print(laten)

            if data.dim() == 1:
                data = data.unsqueeze(1)
            if reconstructed.dim() == 1:
                reconstructed = reconstructed.unsqueeze(1)
            error = torch.mean((data - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())

    return np.array(reconstruction_errors)