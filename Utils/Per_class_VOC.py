import pandas as pd
import numpy as np


# data = pd.read_csv('/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/VOC20_perclass/ResNet32_CE_None_VOC.csv')
# data = pd.read_csv("/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/VOC20_perclass/ResNet32_MG_None_VOC.csv")
# data = pd.read_csv("/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/VOC20_perclass/CSMFF_CE_None_VOC.csv")
data = pd.read_csv("/home/grczh/PycharmProjects/Hierarchical_classification_paper_2/VOC20_perclass/CSMFF_MG_None_VOC.csv")

data.info()
data.head()
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 100)

brother_labels = {0: [6, 5, 1, 13, 3, 18], 1: [6, 5, 13, 1, 0, 3, 18], 2: [7, 9, 11, 12, 16],
                  3: [6, 5, 1, 13, 1, 0, 18], 4: [8, 17, 10, 19, 15], 5: [6, 1, 13, 1, 0, 3, 18],
                  6: [5, 1, 13, 1, 0, 3, 18], 7: [2, 7, 9, 11, 12, 16], 8: [4, 8, 17, 10, 19, 15],
                  9: [2, 7, 9, 11, 12, 16],
                  10: [4, 8, 17, 19, 15], 11: [2, 7, 9, 12, 16], 12: [2, 7, 9, 11, 16], 13: [6, 5, 1, 0, 3, 18],
                  14: [14],
                  15: [4, 8, 17, 10, 19], 16: [2, 7, 9, 11, 12], 17: [4, 8, 10, 19, 15], 18: [6, 5, 1, 13, 0, 3],
                  19: [4, 8, 17, 10, 15]}

err = {'real_num': [], 'pred_err': [], 'brother_error': []}  # 设置字典，键值和键的列表
err_predict = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12: {}, 13: {},
               14: {}, 15: {}, 16: {}, 17: {}, 18: {}, 19: {}}  # 双重字典
err_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
count1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(20):
    d = data.loc[data['test_labels'] == i]
    err['real_num'].append(len(d))

    err_pred = d.loc[data['pred_labels'] != i]

    err['pred_err'].append(len(err_pred))

    err_pred_list = err_pred['pred_labels'].tolist()
    for c in err_pred_list:
        if c in brother_labels[i]:
            count1[i] = count1[i] + 1
    err['brother_error'].append(count1[i])

    # print(err['brother']) #[12, 11, 6, 9, 13, 8, 10, 13, 13, 20, 5, 34, 16, 7, 0, 7, 12, 13, 11, 9]

    # print(count1)#[12, 11, 6, 9, 13, 8, 10, 13, 13, 20, 5, 34, 16, 7, 0, 7, 12, 13, 11, 9]
    err_pred_list = np.unique(err_pred['pred_labels']).tolist()

    for c in err_pred_list:
        # err_predict[i][c] = len(err_pred.loc[err_pred['pred_labels']==c]
        err_count[c] += len(err_pred.loc[err_pred['pred_labels'] == c])
err = pd.DataFrame(err)
err['brother_percent'] = err['brother_error'] / err['real_num']
err['err_percent'] = err['pred_err'] / err['real_num']
err['right_percent'] = (err['real_num'] - err['pred_err']) / err['real_num']

# err  #有一个err就行啦

columns_list = err.columns.tolist()
# print(columns_list) #['real', 'pred_err', 'brother_error', 'brother_percent', 'err_percent', 'right_percent']
columns_list.insert(0, 'name')
# err['name'] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtalbe',
#                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

err['name'] = ['chair', 'bottle', 'sofa', 'diningtable', 'pottedplant', 'tvmonitor', 'train', 'bicycle', 'bus', 'aeroplane', 'motorbike', 'boat', 'car', 'person', 'sheep', 'bird', 'dog', 'cow', 'cat', 'horse']

err = err.reindex(columns=columns_list)

# 放哪个都可以#err['name'] = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtalbe','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
print(err)
print('总的错误', sum(err['pred_err']))
print('兄弟结点错误', sum(err['brother_error']))

err = err.sort_values(by='real_num', ascending=False)
print(err)
err.reset_index(drop=True,inplace=True)
print(err)

head_class = np.arange(0,11,1)
middle_class = np.arange(11,19,1)
tail_class = np.arange(19,20,1)
print(len(head_class),len(middle_class),len(tail_class))

head_err,head_num =0,0

middle_err,middle_num =0,0
tail_err,tail_num =0,0
for i in head_class:
    head_err += err['pred_err'][i]
    head_num +=err['real_num'][i]
for i in middle_class:
    middle_err += err['pred_err'][i]
    middle_num += err['real_num'][i]
for i in tail_class:
    tail_err += err['pred_err'][i]
    tail_num += err['real_num'][i]
print('head_err and middle_err and tail_err',head_err,middle_err,tail_err)
print('head_num and middle_num and tail_num',head_num,middle_num,tail_num)
print("head_acc and middle_acc and tail_acc",1-(head_err/head_num),1-(middle_err/middle_num),1-(tail_err/tail_num))
print('Total acc:', 1-((head_err+middle_err+tail_err)/(head_num+middle_num+tail_num)))

print(err['name'])

# tail_class = [1, 4, 5, 8, 9, 10, 13, 15, 17, 19]
# # tail_class =[5,10,13,15,17]
# a = 0
# b = 0
# for i in tail_class:
#     a += err['pred_err'][i]
#     b += err['brother_error'][i]
# print('尾部的错误数', a)
# print('尾部兄弟结点的错误数', b)
# print('尾部兄弟结点错误所占的比例', b / sum(err['real_num']))
# print('尾部错误所占的比例', a / sum(err['real_num']))
# print('error', sum(err['pred_err']) / sum(err['real_num']))
# print(sum(err['real_num']))

