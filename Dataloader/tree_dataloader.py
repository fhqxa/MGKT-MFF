import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class DatasetLoader(Dataset):

    def __init__(self, setname, return_path=False, data_dir=None):

        # DATASET_DIR = os.path.join(data_dir, 'cifar_100_long_tail_fen')
        DATASET_DIR = data_dir
        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
        else:
            raise ValueError('Unkown setname.')

        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path

        fine_folders = [os.path.join(coarse_label, label) \
                        for coarse_label in coarse_folders \
                        if os.path.isdir(coarse_label) \
                        for label in os.listdir(coarse_label)
                        ]
        # print(coarse_folders)
        # print(fine_folders)

        coarse_labels = np.array(range(len(coarse_folders)))
        coarse_labels = dict(zip(coarse_folders, coarse_labels))
        # print(coarse_labels)
        # fine labels
        labels = np.array(range(len(fine_folders)))
        labels = dict(zip(fine_folders, labels))
        # print('fine_folders',fine_folders)
        # print(labels)

        data = []
        for c in fine_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            data += temp

        # coarse labels
        coarse_label = [coarse_labels['/' + self.get_coarse_class(x)] for x in data]
        #print(coarse_label)

        # fine label
        fine_label = [labels['/' + self.get_class(x)] for x in data]
        print(len(fine_label))

        self.data = data
        self.coarse_label = coarse_label
        self.label = fine_label
        self.num_fine_class = len(set(fine_label))
        self.num_coarse_class = len(set(coarse_label))

        # Transformation
        image_size = 32
        if setname == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            # transforms.Normalize(np.array([x / 255.0 for x in [0.5071, 0.4866, 0.4409]]),
            #                        np.array([x / 255.0 for x in [0.2675, 0.2565, 0.2761]]))])
            # transforms.Normalize(np.array([x / 255.0 for x in [0.5071, 0.4866, 0.4409]]),
            #                      np.array([x / 255.0 for x in [0.2675, 0.2565, 0.2761]]))])

        elif setname == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            # transforms.Normalize(np.array([x / 255.0 for x in [0.5071, 0.4866, 0.4409]]),
            #                      np.array([x / 255.0 for x in [0.2675, 0.2565, 0.2761]]))])

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def get_coarse_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, coarse_label, label = self.data[i], self.coarse_label[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        # print(label,coarse_label) #8 1  #11 2
        return image, label, coarse_label


if __name__ == '__main__':
    train_data_path = '/dataset'
    train_data = tree_cifar_Hier.DatasetLoader('train', return_path=False, data_dir=train_data_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                               shuffle=True, num_workers=4, pin_memory=True)
    # for img,label,coarse_labels in train_loader:
    #     # print(img.shape)
    #     # print('fine',label)
    #     # print('coarse',coarse_labels)
