import pandas as pd
import numpy as np

# data = pd.read_csv(
#     '/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/Cifar100_perclass/CE_None_cifar50.csv')
data = pd.read_csv('/home/grczh/PycharmProjects/RSG-main/Imbalanced_Classification/checkpoint/cifar_resnet32_CE_None_exp_200_0/ckpt.best.pth.tar')
data.info()
data.head(100)
# pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# #设置数据的显示长度，默认为50
# pd.set_option('max_colwidth',200)
# #禁止自动换行(设置为Flase不自动换行，True反之)
# pd.set_option('expand_frame_repr', False)

err = {'real_num': [], 'pred_err': []}
err_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(err_count)

for i in range(100):
    d = data.loc[data['test_labels'] == i]
    err['real_num'].append(len(d))

    err_pred = d.loc[data['pred_labels'] != i]
    err['pred_err'].append(len(err_pred))

    err_pred_list = err_pred['pred_labels'].tolist()
#     for c in err_pred_list:
#         if c in brother_labels[i]:
#             count1[i] =count1[i]+1
# #     err['brother_error'].append(count1[i]) #得到0的兄弟节点加入到字典中的列表中

err = pd.DataFrame(err)  # 把字典列表形成数据表格形式
err['err_percent'] = err['pred_err'] / err['real_num']
# print(err)

columns_list = err.columns.tolist()
# print(columns_list) #['real', 'pred_err', 'brother_error', 'brother_percent', 'err_percent', 'right_percent']
columns_list.insert(0, 'name')
err['name'] = ['衣柜', '床', '椅子', '沙发', '桌子', '平原', '云', '海', '森林', '山', '女人', '女孩', '宝贝', '男孩', '男人', '电话', '电视机', '时钟',
               '台灯', '键盘', '蜜蜂', '蝴蝶', '蟑螂', '毛毛虫', '甲虫', '向日葵', '郁金香', '兰花', '玫瑰', '罂粟花', '老鼠', '仓鼠', '地鼠', '兔子', '松鼠',
               '狐狸', '浣熊', '负鼠', '豪猪', '臭鼬', '鳐', '水族馆鱼', '比目鱼', '鲨鱼', '鳟鱼', '碗', '杯子', '罐', '盘子', '瓶子', '棕榈树', '松树',
               '枫树', '橡树', '柳树', '鳄鱼', '恐龙', '乌龟', '蜥蜴', '蛇', '黑猩猩', '袋鼠', '骆驼', '牛', '大象', '狮子', '豹', '熊', '老虎', '狼',
               '房子', '路', '摩天大楼', '桥', '城堡', '龙虾', '螃蟹', '蜗牛', '蜘蛛', '蠕虫', '鲸鱼', '海豚', '海豹', '海狸', '水獭', '有轨电车', '割草机',
               '火箭', '拖拉机', '坦克', '苹果', '蘑菇', '梨', '甜辣椒', '橘子', '火车', '自行车', '皮卡车', '摩托车', '公共汽车']
err = err.reindex(columns=columns_list)
# print(err)

Train_num_sort = ['苹果', '水族馆鱼', '宝贝', '熊', '海狸', '床', '蜜蜂', '甲虫', '自行车', '瓶子', '碗', '男孩', '桥',
                  '公共汽车', '蝴蝶', '骆驼', '罐', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟', '云', '蟑螂', '沙发',
                  '螃蟹', '鳄鱼', '杯子', '恐龙', '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠', '房子', '袋鼠',
                  '键盘', '台灯', '割草机', '豹', '狮子', '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠', '蘑菇',
                  '橡树', '橘子', '兰花', '水獭', '棕榈树', '梨', '皮卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
                  '负鼠', '兔子', '浣熊', '鳐', '路', '火箭', '玫瑰', '海', '海豹', '鲨鱼', '地鼠', '臭鼬',
                  '摩天大楼', '蜗牛', '蛇', '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜辣椒', '桌子', '坦克', '电话', '电视机',
                  '老虎', '拖拉机', '火车', '鳟鱼', '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫']
test_labels = ['衣柜', '床', '椅子', '沙发', '桌子', '平原', '云', '海', '森林', '山', '女人', '女孩', '宝贝', '男孩', '男人', '电话', '电视机', '时钟',
               '台灯', '键盘', '蜜蜂', '蝴蝶', '蟑螂', '毛毛虫', '甲虫', '向日葵', '郁金香', '兰花', '玫瑰', '罂粟花', '老鼠', '仓鼠', '地鼠', '兔子', '松鼠',
               '狐狸', '浣熊', '负鼠', '豪猪', '臭鼬', '鳐', '水族馆鱼', '比目鱼', '鲨鱼', '鳟鱼', '碗', '杯子', '罐', '盘子', '瓶子', '棕榈树', '松树',
               '枫树', '橡树', '柳树', '鳄鱼', '恐龙', '乌龟', '蜥蜴', '蛇', '黑猩猩', '袋鼠', '骆驼', '牛', '大象', '狮子', '豹', '熊', '老虎', '狼',
               '房子', '路', '摩天大楼', '桥', '城堡', '龙虾', '螃蟹', '蜗牛', '蜘蛛', '蠕虫', '鲸鱼', '海豚', '海豹', '海狸', '水獭', '有轨电车', '割草机',
               '火箭', '拖拉机', '坦克', '苹果', '蘑菇', '梨', '甜辣椒', '橘子', '火车', '自行车', '皮卡车', '摩托车', '公共汽车']

# for c in Train_num_sort:
#
# for i, c in enumerate(Train_num_sort):
#     if c in err['name'][i]:
#         print(i)
#         columns_list = err.sort_values('i')
#         print(columns_list)

# result = err.sort_values(by= 'name', Train_num_sort)
# print(result)

err['name'] = err['name'].astype('category')
err['name'].cat.reorder_categories(Train_num_sort,inplace = True)
err.sort_values('name',inplace=True)
print(err)
#print(err['name'][1])  #bed
print(err.index.values)
err.reset_index(drop=True,inplace=True)
print(err)

head_class = np.arange(0,41,1)
print(head_class)
middle_class = np.arange(41,81,1)
tail_class = np.arange(81,100,1)
print(len(head_class),len(middle_class),len(tail_class))
head_err =0

middle_err =0
tail_err =0
for i in head_class:
    head_err += err['pred_err'][i]
for i in middle_class:
    middle_err += err['pred_err'][i]
for i in tail_class:
    tail_err += err['pred_err'][i]
print('head_err and middle_err and tail_err',head_err,middle_err,tail_err)
print('head_class accuracy',1-head_err/(len(head_class)*100))
print('middle_class accuracy',1-middle_err/(len(middle_class)*100))
print('tail_class accuracy',1-tail_err/(len(tail_class)*100))

print('All accuracy',1-sum(err['pred_err'])/10000)