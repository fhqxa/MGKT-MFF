import torch
import os
flag = 7
if flag==0:
    train_root_path = 'Voc20_datasets/VOC2012_Hier/train'
elif flag==1:
    train_root_path = 'cifar_100_long_tail_fen_ratio_10/train'
elif flag == 2:
    train_root_path = 'cifar_100_long_tail_fen_ratio_20/train'
elif flag == 3:
    train_root_path = 'cifar_100_long_tail_fen_ratio_50/train'
elif flag == 4:
    train_root_path = 'cifar_100_long_tail_fen_ratio_100/train'
elif flag ==5:
    train_root_path = 'cifar_100_long_tail_fen_ratio_200/train'
elif flag ==6:
    train_root_path = 'SUN_datasets/Sun_hier/train'
else:
    train_root_path = 'TiredImagenet_datasets/TiredImagnet_hier/train'

coarse_sum_list = []
for i in os.listdir(train_root_path):
    # print(i)
    # print(len(i))
    coarse_fine_num = []
    for fine in os.listdir(os.path.join(train_root_path,i)):
        coarse_fine_num.append(len(os.listdir(os.path.join(train_root_path,i,fine))))
    coarse_sum = sum(coarse_fine_num)
    coarse_sum_list.append(coarse_sum)
    # print(coarse_fine_num)
coarse_sum_list.sort(reverse=True)
print(len(coarse_sum_list),sum(coarse_sum_list),coarse_sum_list)

# voc = [1508, 1098, 456, 375]
#cifar-100-10 = 20 19572 [1833, 1526, 1428, 1307, 1164, 1137, 1112, 1001, 989, 981, 953, 901, 837, 830, 659, 648, 647, 630, 499, 490]
# cifar-100-20 = 20 15907 [1678, 1349, 1215, 1150, 1003, 963, 921, 835, 824, 821, 792, 685, 635, 630, 473, 444, 442, 424, 319, 304]
# cifar-100-50 = 20 12607 [1498, 1160, 1001, 986, 852, 802, 739, 697, 696, 674, 655, 484, 451, 450, 316, 274, 271, 255, 181, 165]
# cifar-100-100= 20 10847 [1378, 1043, 915, 845, 765, 710, 636, 634, 633, 599, 584, 377, 354, 354, 240, 192, 191, 173, 121, 103]
# cifar-100-200= 20 9502 [1269, 942, 846, 726, 692, 638, 589, 589, 554, 542, 533, 298, 283, 279, 185, 137, 136, 117, 82, 65]
# sun = 15 85147 [12966, 8720, 6809, 6661, 6408, 6294, 5690, 5638, 4946, 4507, 4125, 4034, 3057, 3030, 2262]
#tired= 34 69545 [4486, 3936, 3671, 3556, 3517, 3436, 3135, 2746, 2513, 2453, 2370, 2366, 2335, 2131, 2117, 1970, 1937, 1830, 1784, 1668, 1507, 1465, 1395, 1271, 1257, 1253, 1251, 1243, 1125, 1055, 893, 675, 637, 561]
