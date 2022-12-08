import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
  def __init__(self, imgs_path=str, transforms=None):
    self.imgs_path = imgs_path
    self.transforms = transforms
    file_list = glob.glob(self.imgs_path + "*")
    self.data = []
    for class_path in file_list:
      class_name = class_path.split("/")[-1]
      for img_path in glob.glob(class_path + "/*.jpg"):
        self.data.append([img_path, class_name])
    self.class_map = {"open" : 0, "close": 1}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    img = cv2.imread(img_path, 1)
    #Applying transforms on image
    if self.transforms:
      img = self.transforms(image=img)["image"]
    # img = torch.from_numpy(img)
    # img = img.permute(2, 0, 1)
    label = self.class_map[class_name]

    return img, label

if __name__ == "__main__":
  dataset = CustomDataset(imgs_path='./data_final/val/')		
  data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
  total_imgs = 0
  for imgs, labels in data_loader:
    total_imgs += int(imgs.shape[0])
  print(total_imgs)
    # print("Batch of images has shape: ",imgs.shape)
    # print("Batch of labels has shape: ", labels.shape)
    