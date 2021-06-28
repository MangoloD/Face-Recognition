import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, file_path, doc_path, flag, is_mark=True):
        self.path = f"{file_path}"
        self.dataset = []
        self.is_mark = is_mark
        if flag == 0:
            self.dataset.extend(open(f"{file_path}/{doc_path[0]}").readlines())
            self.dataset.extend(open(f"{file_path}/{doc_path[1]}").readlines())
            self.dataset.extend(open(f"{file_path}/{doc_path[2]}").readlines())
        elif flag == 1:
            self.dataset.extend(open(f"{file_path}/{doc_path[0]}").readlines())
            self.dataset.extend(open(f"{file_path}/{doc_path[1]}").readlines())
            self.dataset.extend(open(f"{file_path}/{doc_path[2]}").readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].strip().split()
        img_path = f"{self.path}/{data[0]}"
        con = torch.Tensor([int(data[1])])
        if self.is_mark is False:
            loc = torch.Tensor([float(data[2]), float(data[3]), float(data[4]), float(data[5])])
        else:
            loc = torch.Tensor([float(data[2]), float(data[3]), float(data[4]), float(data[5]),
                                float(data[6]), float(data[7]), float(data[8]), float(data[9]),
                                float(data[10]), float(data[11]), float(data[12]), float(data[13]),
                                float(data[14]), float(data[15])])
        img_data = self.train_transform(Image.open(img_path))
        return img_data, con, loc


if __name__ == '__main__':
    root_path = ["G:/MTCNN_Data/Dataset_1/Train_Image", "G:/MTCNN_Data/Dataset_1/Validate_Image",
                 "G:/MTCNN_Data/Dataset_1/Test_Image"]
    pic_size_l = [12, 24, 48]
    doc_name = [["train_positive.txt", "train_part.txt", "train_negative.txt"],
                ["validate_positive.txt", "validate_part.txt", "validate_negative.txt"],
                ["test_positive.txt", "test_part.txt", "test_negative.txt"]]

    filepath = f"{root_path[0]}/{str(pic_size_l[2])}"
    dataset = FaceDataset(filepath, doc_name[0], 1, False)
    # print(dataset[1][0].shape)
    # print(dataset[1][1].shape)
    # print(dataset[1][2].shape)
    dataloader = DataLoader(dataset, 5, shuffle=True, num_workers=1)
    for i, (img, cls, offset) in enumerate(dataloader):
        print(img.shape)
        print(cls.shape)
        # print(cls)
        print(offset.shape)
