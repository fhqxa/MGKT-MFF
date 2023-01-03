import os
import os.path as osp

# # -------------------------------- compute per_class_num ----------------------------------------
# train_data_path = '/dataset'
# def per_class_num():
#     THE_PATH = os.path.join(train_data_path, 'cifar_100_long_tail_fen', 'train')
#     coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
#                       os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path
#     fine_folders = [os.path.join(coarse_label, label) \
#                     for coarse_label in coarse_folders \
#                     if os.path.isdir(coarse_label) \
#                     for label in os.listdir(coarse_label)
#                     ]
#     per_num_class_fine = []
#     per_num_class_coarse = []
#     for path in fine_folders:
#         #print(path)
#         a = len(os.listdir(path))
#         per_num_class_fine.append(a)
#     print(per_num_class_fine)
#     print(len(per_num_class_fine))
#
#     for path in coarse_folders:
#         #print(path)
#         b = len(os.listdir(path))
#         per_num_class_coarse.append(b)
#     print(per_num_class_coarse)
#
#
#     # print(coarse_folders)
#     # print(fine_folders)
# per_class_num()


# train_root_path = '/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/iNat_datasets/iNat_Flat/train'
# test_root_path ='/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/iNat_datasets/iNat_Flat/test'

train_root_path = "/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/SUN_datasets/SUN_Flat/SUN397/train"
test_root_path = "/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/SUN_datasets/SUN_Flat/SUN397/test"

# train_root_path = "/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/TiredImagenet_datasets/TiredImageNet_Flat/tieredImageNet/train"
# test_root_path = "/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/Long-tailed datasets/TiredImagenet_datasets/TiredImageNet_Flat/tieredImageNet/test"
inat_train = len(os.listdir(train_root_path))
inat_test = len(os.listdir(test_root_path))
print(inat_train)
print(inat_test)
train_num_per = []
for i in os.listdir(train_root_path):
    num = len(os.listdir(os.path.join(train_root_path, i)))
    train_num_per.append(num)
print(train_num_per)
train_num_per.sort(reverse=True)
print('sort',train_num_per)
print(sum(train_num_per))
print('iNaturalist_train:', train_num_per[0] / train_num_per[323])

test_num_per = []
for i in os.listdir(test_root_path):
    num = len(os.listdir(os.path.join(test_root_path, i)))
    test_num_per.append(num)
print(test_num_per)
test_num_per.sort(reverse=True)
print(test_num_per)
print(sum(test_num_per))
print('iNaturalist_test:', test_num_per[0] / test_num_per[323])

