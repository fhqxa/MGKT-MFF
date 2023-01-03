from PIL import Image
import os
import shutil
import pickle
from matplotlib import patches
import numpy as np

from tqdm import trange

# create image directory #100类中对应粗类
flag_create_image_directory = True
if flag_create_image_directory:
    relation_f1 = '''
4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11,
1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12,
16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14,
13  
'''


# str1 = '''1 2 3 4 5 ''' or '''1, 2, 3, 4'''  there is a blank back #目的把str转为列表
def StrTolist_relation(str1):
    list1 = []
    t = ''
    for i in str1:
        if i == '\n':
            continue
        if i != ' ' and i != ',':
            t += i
        else:
            if t != '':
                list1.append(int(t))
                t = ''
    return list1


relation_f = StrTolist_relation(relation_f1)


def CreateImageDirectory(isCorse=False, long_tail=False, isChinese=False, ratio_longtail=100):
    # long_tail1 = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206,
    # 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61,
    # 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15,
    # 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]
    relation_f1 = '''
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11,
    1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12,
    16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14,
    13  
    '''
    list1 = []
    t = ''
    for i in relation_f1:
        if i == '\n':
            continue
        if i != ' ' and i != ',':
            t += i
        else:
            if t != '':
                list1.append(int(t))
                t = ''
    relation_f = list1


    #flat and hier test dataset ensure
    path_dataset = '/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/dataset/'
    #name_directory = '/cifar_100_long_tail_fen_ratio_200/train/'  # ./当前路径
    name_directory = 'Long-tailed datasets/Cifar100_datasets/cifar_100_long_tail_fen/cifar_100_long_tail_fen_ratio_20/train/'  # ./当前路径
    long_tail1 = Cifar100_LongTailDistribution(ratio_longtail)
    path_data = path_dataset + 'cifar-100-python'
    path_directory = path_dataset + name_directory
    with open(path_data + '/meta', 'rb') as fo:
        dict_meta = pickle.load(fo, encoding='latin1')

    with open(path_data + '/train', 'rb') as fo:
        dict_train = pickle.load(fo, encoding='latin1')

    shutil.rmtree(path_directory) if os.path.exists(path_directory) else ''
    os.makedirs(path_directory) if not os.path.exists(path_directory) else ''

    name_f_classes = dict_meta['fine_label_names']
    name_c_classes = dict_meta['coarse_label_names']

    if isChinese:
        name_c_classes_ = ['水生哺乳动物', '鱼', '花卉', '食品容器', '水果和蔬菜', '家用电器', '家庭家具', '昆虫', '大型食肉动物',
                           '大型人造户外用品', '大自然户外场景', '大型杂食动物和食草动物', '中型哺乳动物', '非昆虫无脊椎动物',
                           '人', '爬行动物', '小型哺乳动物', '树木', '车辆一', '车辆二']

        name_f_classes_ = ['苹果', '水族馆鱼', '宝贝', '熊', '海狸', '床', '蜜蜂', '甲虫', '自行车', '瓶子', '碗', '男孩', '桥',
                           '公共汽车', '蝴蝶', '骆驼', '罐', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟', '云', '蟑螂', '沙发',
                           '螃蟹', '鳄鱼', '杯子', '恐龙', '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠', '房子', '袋鼠',
                           '键盘', '台灯', '割草机', '豹', '狮子', '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠', '蘑菇',
                           '橡树', '橘子', '兰花', '水獭', '棕榈树', '梨', '皮卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
                           '负鼠', '兔子', '浣熊', '鳐', '路', '火箭', '玫瑰', '海', '海豹', '鲨鱼', '地鼠', '臭鼬',
                           '摩天大楼', '蜗牛', '蛇', '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜辣椒', '桌子', '坦克', '电话', '电视机',
                           '老虎', '拖拉机', '火车', '鳟鱼', '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫']

    else:
        name_f_classes_ = name_f_classes
        name_c_classes_ = name_c_classes

    for i_f in range(len(name_f_classes)):
        i_c = relation_f[i_f]
        i_c_name = name_c_classes_[i_c]  # 英文
        i_f_name = name_f_classes_[i_f]

        if isCorse:
            path = path_directory + f'{i_c_name}/' + f'{i_f_name}/'
        else:
            path = path_directory + f'{i_f_name}/'
        os.makedirs(path) if not os.path.exists(path) else ''

    data = dict_train['data']  # (50000,3072)
    labels = dict_train['fine_labels']  # 50000
    if long_tail:  # bool

        count = np.zeros(len(name_f_classes), dtype=np.int64)
        new_labels = []
        for i in trange(len(labels)):
            if count[labels[i]] < long_tail1[labels[i]]:
                count[labels[i]] += 1

                if i == 0:
                    new_data = data[i].reshape(1, -1)
                else:
                    new_data = np.concatenate((new_data, data[i].reshape(1, -1)))
                new_labels.append(labels[i])
            else:
                continue  # 继续for i 的循环

        data = new_data  # 更新长尾的图片数据
        labels = new_labels  # 更新长尾的标签

    for i in trange(data.shape[0]):
        img = np.reshape(data[i], (3, 32, 32))  # numpy 中把数据转为三维矩阵以便保存为图片
        i0 = Image.fromarray(img[0])
        i1 = Image.fromarray(img[1])
        i2 = Image.fromarray(img[2])
        img = Image.merge('RGB', (i0, i1, i2))

        i_f = labels[i]
        i_c = relation_f[i_f]

        i_c_name = name_c_classes_[i_c]
        i_f_name = name_f_classes_[i_f]

        if isCorse:
            path = path_directory + f'{i_c_name}/' + f'{i_f_name}/'
        else:
            path = path_directory + f'{i_f_name}/'
        img.save(path + dict_train['filenames'][i])


def Unblance_cifar10_folder(ratio_longtail=100):
    train_root_path = '/home/grczh/zwzw/数据集/CIFAR-10-dataset/train'

    root_path = '/home/grczh/zwzw/数据集/'
    dataset = "cifar10_unblance"
    mode = "train"
    cifar10_unblance_folder = root_path + dataset + '_' + mode
    shutil.rmtree(cifar10_unblance_folder) if os.path.exists(cifar10_unblance_folder) else os.makedirs(
        cifar10_unblance_folder)
    Longtail_num_per = Cifar10_LongTailDistribution(ratio_longtail)
    filesnames = os.listdir(train_root_path)
    filesnames.sort(key=lambda x: str(x.split('.')[0]))
    print('filesnames',
          filesnames)  # filesnames ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i, label in enumerate(filesnames):
        os.makedirs(os.path.join(cifar10_unblance_folder, label))
        per_num = 0
        print('label', label)
        for img in os.listdir(os.path.join(train_root_path, label)):
            img1 = os.path.join(train_root_path, label, img)
            # print(img)
            if per_num < Longtail_num_per[i]:
                img1 = Image.open(img1)
                img1.save(os.path.join(cifar10_unblance_folder, label, img))
                per_num += 1
            else:
                break


def Cifar10_LongTailDistribution(ratio_longtail=100):
    num_perclass_train = [5000] * 10
    max_num = max(num_perclass_train)
    class_num = len(num_perclass_train)
    mu = np.power(1 / ratio_longtail, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        # print('%.2f' % np.power(mu, i), end=' ')
        # 1.00 0.60 0.36 0.22 0.13 0.08 0.05 0.03 0.02 0.01
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def Cifar100_LongTailDistribution(ratio_longtail=100):
    num_perclass_train = [500] * 100
    max_num = max(num_perclass_train)
    class_num = len(num_perclass_train)
    mu = np.power(1 / ratio_longtail, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        # print('%.2f' % np.power(mu, i), end=' ')
        # 1.00 0.60 0.36 0.22 0.13 0.08 0.05 0.03 0.02 0.01
        class_num_list.append(int(max_num * np.power(mu, i)))
    # print(list(class_num_list)) #[500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, ...
    return list(class_num_list)


def Voc_LongTailDistribution():
    data_root_path_train = '/home/grczh/zwzw/数据集/VOC2012_原始/train'
    filesnames = os.listdir(data_root_path_train)
    filesnames.sort(key=lambda x: str(x.split('.')[0]))  # 加不加这个差别很大
    class_num_list = []
    for i in filesnames:
        # print(i)
        num = len(os.listdir(os.path.join(data_root_path_train, i)))
        class_num_list.append(num)
    # print(class_num_list)
    return class_num_list


def tinyimagenet_LongTailDistribution(ratio_longtail=10):
    num_perclass_train = [500] * 200
    max_num = max(num_perclass_train)
    class_num = len(num_perclass_train)
    mu = np.power(1 / ratio_longtail, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        # print('%.2f' % np.power(mu, i), end=' ')
        # 1.00 0.60 0.36 0.22 0.13 0.08 0.05 0.03 0.02 0.01
        class_num_list.append(int(max_num * np.power(mu, i)))
    # print(class_num_list)
    # print(len(class_num_list))
    return list(class_num_list)


#CreateImageDirectory(isCorse=True, long_tail=True, isChinese=True, ratio_longtail=20)
# Voc_LongTailDistribution()
# Cifar100_LongTailDistribution()
