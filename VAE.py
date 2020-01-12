import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

REBUILD_DATA = False
IMG_SIZE = 120
LATENT_DIM = 2

if not os.path.exists("models/"):
    os.mkdir("models")


class DataBuilder():
    file_dir = "images/"

    def pngTojpg(self):
        for image in tqdm(os.listdir(self.file_dir)):
            PnGpath = os.path.join(self.file_dir, image)
            img = cv2.imread(PnGpath)
            new_path = PnGpath.replace(".png", ".jpg")
            cv2.imwrite(new_path, img)
            os.remove(PnGpath)

    def make_data(self):
        all_data = []
        for image in tqdm(os.listdir(self.file_dir)):
            path = os.path.join(self.file_dir, image)
            img = cv2.imread(path)
            if len(img[0]) == IMG_SIZE:
                all_data.append(img)
                #scaled = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        all_data = np.array(all_data)
        np.save("data.npy", all_data)

if REBUILD_DATA:
    DataBuilder().make_data()
    
data = np.load("data.npy")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.linear = 3136
        
        self.fc1 = nn.Linear(self.linear, 256)
        self.fc2 = nn.Linear(256, LATENT_DIM)
        
        # Decoder part
        self.fc3 = nn.Linear(LATENT_DIM, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, IMG_SIZE**2*3)


    def conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)

        self._to_linear = x.shape[1]*x.shape[2]*x.shape[3]
        assert self._to_linear == self.linear, f"Please set self.linear to {self._to_linear}"

        x = x.view(-1, self.linear)

        return x

    def encode(self, x):
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def decode(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return torch.sigmoid(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x.view(-1, 3, 120, 120)

    def getVectorSpace(self, x):
        return self.encode(x)

    def getImageFromVectorSpace(self, z):
        return self.decode(z).view(IMG_SIZE, IMG_SIZE, 3)

# testing the NN



data = torch.tensor([i for i in data]).view(-1, IMG_SIZE, IMG_SIZE, 3).to(device)
data = data / 255.0

vae = VAE().to(device)



import matplotlib.pyplot as plt

with torch.no_grad():
    r = torch.randn((1, 3, IMG_SIZE, IMG_SIZE)).to(device)
    x = vae(r)

    x = x.view(-1, 3, IMG_SIZE,IMG_SIZE)
    print(x.shape)


    #plt.imshow(x[0].view(IMG_SIZE, IMG_SIZE, 3))
    #plt.show()

EPOCHS = 10000
BATCH_SIZE = 256

loss_over_time = np.empty(EPOCHS)

import time

directory = str(round(time.time()))

def train():

    opt = optim.Adam(vae.parameters(), lr=0.001)
    lossFunc = nn.MSELoss()

    for epoch in range(EPOCHS):
        train_loss = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:BATCH_SIZE+i].view(-1, 3, IMG_SIZE, IMG_SIZE)

            opt.zero_grad()
            outputs = vae(batch)

            train_loss = lossFunc(outputs, batch)
            train_loss.backward()
            opt.step()


        #file_name = f"epoch{epoch}.jpg"
        #if epoch == 1:
            #plt.imsave(f"{directory}/original.jpg", batch[imginx].view(IMG_SIZE, IMG_SIZE, 3).detach().cpu())
            #plt.imshow(batch[imginx].view(IMG_SIZE, IMG_SIZE, 3).detach().cpu())
            #plt.show()
        #plt.imsave(os.path.join(directory, file_name), outputs[imginx].view(IMG_SIZE, IMG_SIZE, 3).detach().cpu())
        loss_over_time[epoch] = train_loss.item()

        if epoch % 100 == 0:
            print(f"{epoch} epochs Completed...")

    torch.save(vae.state_dict(), f"models/{directory}_latentdim-{LATENT_DIM}.pt")
    np.save(f"loss-{directory}.npy", loss_over_time)




def show_pokemon_image(modelname, pokemon):
    vae = VAE()
    vae.load_state_dict(torch.load(f"models/{modelname}"))
    with torch.no_grad():
        try:
            img = cv2.imread(f"images/{pokemon}.jpg")
        except FileNotFoundError as e:
            print(e)

        img = torch.tensor(img).view(-1, 3, IMG_SIZE, IMG_SIZE)
        img = img / 255.0

        vae_pokemon = vae(img)
        vae_pokemon = vae_pokemon.view(IMG_SIZE, IMG_SIZE, 3)

        #print(vae.getVectorSpace(img))

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)

        ax1.imshow(img.view(IMG_SIZE, IMG_SIZE, 3))
        ax2.imshow(vae_pokemon)

        plt.show()

import matplotlib
from matplotlib import style
from scipy.cluster.vq import kmeans
from scipy import spatial

style.use("ggplot")
matplotlib.use('TkAgg')
def make_3d_dimensionallity_space(modelname):

    img_dir = "images/"
    vae = VAE()
    vae.load_state_dict(torch.load(f"models/{modelname}"))

    all_image_vector_data = {}

    with torch.no_grad():
        for i, img in enumerate(tqdm(os.listdir(img_dir))):

            file_name = img

            img = cv2.imread(os.path.join(img_dir, img))
            img = torch.tensor(img).view(-1, 3, IMG_SIZE, IMG_SIZE)
            img = img / 255.0

            img_vector = vae.getVectorSpace(img)
            all_image_vector_data[file_name.replace(".jpg", "")] = img_vector.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        for pokemon in all_image_vector_data:
            vector = all_image_vector_data[pokemon][0]
            #ax.scatter(vector[0], vector[1], vector[2], s=1)
            ax.text(vector[0], vector[1], vector[2], pokemon)

        plt.show()

def make_2d_dimensionallity_space(modelname):

    img_dir = "images/"
    vae = VAE()
    vae.load_state_dict(torch.load(f"models/{modelname}"))

    all_image_vector_data = {}

    with torch.no_grad():
        for i, img in enumerate(tqdm(os.listdir(img_dir))):

            file_name = img
            original_img = img
            img = cv2.imread(os.path.join(img_dir, img))
            img = torch.tensor(img).view(-1, 3, IMG_SIZE, IMG_SIZE)
            img = img / 255.0

            img_vector = vae.getVectorSpace(img)
            all_image_vector_data[file_name.replace(".jpg", "")] = img_vector.numpy()

        print(all_image_vector_data.values())
        centroids, distortion= kmeans(list(map(lambda x: x[0],list(all_image_vector_data.values()))), 6)

        colours = {0: "r", 1: "b", 2: "y", 3: "g", 4: "m", 5: "k", 6: "c"}

        centroids_tree = spatial.KDTree(centroids)
        fig = plt.figure(1)
        ax = fig.gca()
        for pokemon in all_image_vector_data:
            vector = all_image_vector_data[pokemon][0]
            closest_neighbor = centroids_tree.query(vector)[1]
            ax.text(vector[0], vector[1], pokemon, color=colours[closest_neighbor])



def random_pokemon(modelname):
    vae = VAE()
    vae.load_state_dict(torch.load(f"models/{modelname}"))

    fig = plt.figure(2)
    ax = fig.gca()
    with torch.no_grad():
        z = torch.rand(2)
        print(z)
        new_pokemon = vae.getImageFromVectorSpace(z).cpu()
        print(new_pokemon.shape)
        ax.imshow(new_pokemon.numpy())
        ax.title(str(z))



# Run this if you want to train the VAE
train()

# Run this is you want to view a 3D plot of the 3D latent dim
#show_pokemon_image("1578837142_latentdim-2.pt", "conkeldurr")

# Run this is you want to view a 3D plot of the 3D latent dim
#make_2d_dimensionallity_space("1578837142_latentdim-2.pt")

# Run this if you want to create a weird looking pokemon
#random_pokemon("1578837142_latentdim-2.pt")
plt.show()