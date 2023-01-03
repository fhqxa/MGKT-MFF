# def tree_ANcestor(tr, nd):
#     #  tree: 二维数组   node：节点  默认根节点为0
#     A = [nd]  # 存储
#     print(A)
#     # print(nd-1)
#     nd_ = tr[nd - 1]  # python 从 0 开始算，而结点从 1 开始算
#
#     while nd_ > 0:
#         A.append(nd_)
#         nd_ = tr[nd_ - 1]  # 找父结点
#
#     return A
#
# # def tree_ANcestor1(tree,nd):
# #     A =[nd]
# #
#
# # TIE
# def compute_TIE(tr, p_nd, r_nd):
#     TIE = 0
#     for i in range(len(p_nd)):
#         r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
#         p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
#         b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
#         c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
#         TIE = TIE + len(b + c)
#     # r_anc = tree_ANcestor(tr, r_nd)  # 真实标签的父结点
#     # p_anc = tree_ANcestor(tr, p_nd)  # 预测标签的父结点
#     # b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
#     # c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
#     # TIE = len(b + c)
#     return TIE
#
#
# # create tree_array
# def create_array(coarse_list):
#     num_coarse = max(coarse_list) + 1
#     print(num_coarse)
#     tree = [0]
#     for i in range(num_coarse):
#         tree.append(1)
#
#     for x in coarse_list:
#         tree.append(x + 2)
#
#     return tree
#
#
# # FH
# def compute_FH(tr, p_nd, r_nd):
#     sum_PH, sum_RH, sum_FH = 0, 0, 0
#     length = len(p_nd)
#     for i in range(length):
#         r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
#         p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
#         b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集
#
#         PH = len(b) / len(p_anc)
#         RH = len(b) / len(r_anc)
#         FH = 2 * PH * RH / (PH + RH)
#
#         sum_PH = sum_PH + PH
#         sum_RH = sum_RH + RH
#         sum_FH = sum_FH + FH
#
#     PH = sum_PH / length
#     RH = sum_RH / length
#     FH = sum_FH / length
#
#     return FH  # , PH, RH
#
#
# # pred coarse
# def pred_coarse_label(pred_fine_label, sample_coarse_label):
#     coarse_labels = []
#     for x in pred_fine_label:
#         coarse_labels.append(sample_coarse_label[x])
#     return coarse_labels
#
#
# import numpy as np
#
# coarse_list = [1,2,3]
# print(coarse_list)
# tree = create_array(coarse_list)
# print(tree)
#
# # tree = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
# # pred =[1,2]
# # real =[2,3]
# # a =compute_FH(tree,pred,real)
# # print(a)
#


def tree_ANcestor(tr, nd):
    #  tree: 二维数组   node：节点  默认根节点为0

    A = [nd]  # 存储
    nd_ = tr[nd-1]  # python 从 0 开始算，而结点从 1 开始算

    while nd_ > 0:
        A.append(nd_)
        nd_ = tr[nd_-1]  # 找父结点

    return A


# TIE
def EvaHier_TreeInducedError(tr, p_nd, r_nd):
    TIE = 0
    for i in range(len(p_nd)):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
        c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
        TIE = TIE + len(b + c)

    TIE = TIE / len(p_nd)
    return TIE


# FH
def EvaHier_HierarchicalPrecisionAndRecall(tr, p_nd, r_nd):
    sum_PH, sum_RH, sum_FH = 0, 0, 0
    length = len(p_nd)
    for i in range(length):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集

        PH = len(b) / len(p_anc)
        RH = len(b) / len(r_anc)
        FH = 2 * PH * RH / (PH + RH)

        sum_PH = sum_PH + PH
        sum_RH = sum_RH + RH
        sum_FH = sum_FH + FH

    PH = sum_PH / length
    RH = sum_RH / length
    FH = sum_FH / length

    return FH  # , PH, RH


# create tree_array
def create_array(coarse_list):
    num_coarse = max(coarse_list) + 1
    tree = [0]
    for i in range(num_coarse):
        tree.append(1)

    for x in coarse_list:
        tree.append(x + 2)

    return tree



#eg1
# # a = [0,1,1,2,2,2]
# # b =EvaHier_TreeInducedError(a ,[5] ,[3])
# # c =EvaHier_HierarchicalPrecisionAndRecall(a ,[5] ,[3])
# aa =[0,0,0]
# tr = create_array(aa)

# print(b)
# print(c)
# print(tr)

#eg2

# def tree_ANcestor(tr, nd):
#     #  tree: 二维数组   node：节点  默认根节点为0
#
#     A = [nd]  # 存储
#     nd_ = tr[nd - 1]  # python 从 0 开始算，而结点从 1 开始算
#
#     while nd_ > 0:
#         A.append(nd_)
#         nd_ = tr[nd_ - 1]  # 找父结点
#
#     return A
# aa =[0,0,0,1,1]
# print(create_array(aa))
# print(EvaHier_TreeInducedError(aa,[0],[1]))
