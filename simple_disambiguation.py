import json
import numpy as np
# import matplotlib.pyplot as plt
from collections import defaultdict

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 训练集分析
train_row_data_path = 'data/train/train_author.json'
train_pub_data_path = 'data/train/train_pub.json'

train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
train_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))
authors = [author for author in train_data]
authors_num_person = [len(train_data[author].keys()) for author in train_data]

valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'

validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data:
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]

'''
print('训练集同名数量：', len(authors))
print('消歧后实际作者数量：',  sum(authors_num_person))

# 绘制训练集同名作者个体数量
plt.figure(figsize=(40, 8), dpi=80)
x = range(len(authors))

plt.bar(x, authors_num_person, width=0.5)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('训练集同名作者')
plt.ylabel('该名字同名作者数量（个）')
for xl, yl in zip(x, authors_num_person):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

mean_person = int(np.mean(authors_num_person))
plt.gca().hlines(mean_person,-1,225,linestyles='--',colors='red',label='平均值')
plt.annotate(u"平均值:" + str(mean_person), xy = (225, mean_person), xytext = (225, mean_person+40),arrowprops=dict(facecolor='red',shrink=0.1,width=2))

plt.show()

# 绘制训练集同名作者论文总数
authors_num_papers = []
for author in train_data:
    num = 0
    for author_id in train_data[author]:
        papers = train_data[author][author_id]
        num += len(papers)    
    authors_num_papers.append(num)

plt.figure(figsize=(40, 8), dpi=80)
x = range(len(authors))

plt.bar(x, authors_num_papers, width=0.5)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('训练集同名作者')
plt.ylabel('该名字论文总数（篇）')
for xl, yl in zip(x, authors_num_papers):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

mean_person = int(np.mean(authors_num_papers))
plt.gca().hlines(mean_person,-1,225,linestyles='--',colors='red',label='平均值')
plt.annotate(u"平均值:" + str(mean_person), xy = (225, mean_person), xytext = (225, mean_person+40),arrowprops=dict(facecolor='red',shrink=0.1,width=2))

plt.show()

# 合并数据

# 验证集数据分析
authors = validate_data.keys()
papers_perauthor = [len(validate_data[author]) for author in validate_data]
print('同名作者数量：', len(authors))
print('涉及的论文数：', np.sum(papers_perauthor))
print('平均论文数量：', np.mean(papers_perauthor))
print('提供的论文数：',len(validate_pub_data))

# 绘制同名作者论文数量
plt.figure(figsize=(20, 8), dpi=80)
x = range(len(authors))

plt.bar(x, papers_perauthor, width=0.8)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('测试集同名作者')
plt.ylabel('测试集论文数量（篇）')
for xl, yl in zip(x, papers_perauthor):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

plt.gca().hlines(np.mean(papers_perauthor),-1,50,linestyles='--',colors='red',label='平均值')
plt.annotate(u"平均值", xy = (0, np.mean(papers_perauthor)), xytext = (0, 1400),arrowprops=dict(facecolor='red',shrink=0.1,width=2))

plt.show()

# 查看某个同名作者论文情况
author = 'atsushi_takeda'
papers = validate_data[author]
print('消歧作者名：', author)
print('涉及论文数量：', len(papers))

venue_dict = defaultdict(int)
year_dict = defaultdict(int)
keywords_dict = defaultdict(int)
org_dict = defaultdict(int)

for paper in papers:
    authors = paper['authors']
    venues = paper['venue']
    years = paper['year']
    keywords = paper['keywords']

    venue_dict[venues] += 1
    year_dict[years] += 1
    for keyword in keywords:
        keywords_dict[keyword] += 1

    for paper_author in authors:
        name = paper_author['name']
        org =  paper_author['org'] if 'org' in paper_author else ""
        org_dict[org] += 1


# 绘制该名称下论文数据情况
fig = plt.figure(figsize=(20, 20), dpi=80)

ax1 = fig.add_subplot(2,2,1)
x = range(5)
y = [len(papers), len(venue_dict), len(year_dict), len(keywords_dict), len(org_dict)]
s = ['涉及论文数量', '涉及期刊数量', '涉及年份数量', '涉及关键字数量', '涉及机构数量']

plt.bar(x, y, width=0.5)
plt.xticks(x, s, rotation=270)  
plt.xlabel('%s论文数据情况' % author)
plt.ylabel('数量（个）')
for xl, yl in zip(x, y):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

ax2 = fig.add_subplot(2,2,2)
plt.bar(range(len(venue_dict)), venue_dict.values(), width=0.3)
plt.xlabel('%s期刊数据情况' % author)
plt.ylabel('数量（个）')

ax3 = fig.add_subplot(2,2,3)
plt.bar(range(len(year_dict)), year_dict.values(), width=0.5)
plt.xticks(range(len(year_dict)), year_dict.keys(), rotation=270)
plt.xlabel('%s年份数据情况' % author)
plt.ylabel('数量（个）')

ax4 = fig.add_subplot(2,2,4)
plt.bar(range(len(org_dict)), org_dict.values(), width=0.5)  
plt.xlabel('%s机构数据情况' % author)
plt.ylabel('数量（个）')
plt.show()

# 查看论文作者名中是否包含消歧作者名
print(authors)
for author in validate_data:
    print('disambiguation name: ', author)
    papers = validate_data[author]
    for paper in papers[:10]:
        print('\npaper id: ' + paper['id'])
        authors = paper['authors']
        for paper_author in authors:
            name = paper_author['name']
            org = paper_author['org']
            print('paper author name: ', name)
            # print(org)

    break

作者名存在不一致的情况：
1、大小写
2、姓、名顺序不一致
3、下划线、横线
4、简写与不简写
5、姓名有三个字的表达: 名字是否分开

同理：机构的表达也存在不一致的情况
因此：需要对数据做相应的预处理统一表达
'''

import re
# 数据预处理

# 预处理名字
def precessname(name):   
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name)
    return name

# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
    return org

#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''

def disambiguate_by_org():
    res_dict = {}

    for author in validate_data:
        res_dict[author] = []

        print(author)
        papers = validate_data[author]
        org_dict = {}
        org_dict_coauthor = {}
        no_org_list = []
        for paper in papers:
            authors = paper['authors']
            for paper_author in authors:
                name = precessname(paper_author['name'])
                org = preprocessorg(paper_author['org']) if 'org' in paper_author else ""
                name = name.split('_')

                if '_'.join(name) == author or '_'.join(name[::-1]) == author:
                    if org == "":
                        no_org_list.append((paper['id'], [precessname(paper_author['name']) for paper_author in authors]))
                    else:  # 按组织聚类
                        if org not in org_dict:
                            org_dict[org] = [paper['id']]
                            org_dict_coauthor[org] = [precessname(paper_author['name']) for paper_author in authors]
                        else:
                            org_dict[org].append(paper['id'])
                            org_dict_coauthor[org].extend([precessname(paper_author['name']) for paper_author in authors])

        # 没有组织的根据合作者交集
        for p_id, names in no_org_list:
            tmp = ""
            max_num = 1
            for org in org_dict_coauthor:
                set_a = set(names)
                set_b = set(org_dict_coauthor[org])
                intersection = set_a & set_b
                iou = len(intersection)
                if iou > max_num:
                    max_num = iou
                    tmp = org

            if max_num != 1:
                org_dict[tmp].append(p_id)
            else:
                res_dict[author].append([p_id])

        for org in org_dict:
            res_dict[author].append(org_dict[org])
        json.dump(res_dict, open('result/disambiguate_by_org_result.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_org()

# 2. 基于规则（按合作者与组织机构消歧） 线上得分：0.2449
def disambiguate_by_coauthor():
    res_dict = {}

    for author in validate_data:
        print(author)
        res = []
        papers = validate_data[author]
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            d = {}
            authors = [precessname(paper_author['name']) for paper_author in paper['authors']]
            if author in authors:
                authors.remove(author)
            org = preprocessorg(get_org(paper['authors'], author))
            venue = paper['venue']
            d["authors"] = authors
            d["org"] = org
            d['keywords'] = paper['keywords'] if 'keywords' in paper else ""
            d['venue'] = venue

            if len(res) == 0:
                res.append([paper['id']])
            else:
                max_inter = 0
                indx = 0
                for i, clusters in enumerate(res):
                    score = 0
                    for pid in clusters:
                        insection = set(paper_dict[pid]['authors']) & set(authors)
                        score += len(insection)

#                         if org != "" and (org in paper_dict[pid]['org'] or paper_dict[pid]['org'] in org):
#                             score += 10

                    if score > max_inter:
                        max_inter = score
                        indx = i

                if max_inter > 0:
                    res[indx].append(paper['id']) # 若最高分大于0，将id添加到得分最高的簇中
                else:
                    res.append([paper['id']]) # 否则，另起一簇

            paper_dict[paper['id']] = d

        res_dict[author] = res
    json.dump(res_dict, open('result/disambiguate_by_coauthor_result.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_coauthor()


# 3. 无监督聚类（根据合作者和机构TFIDF进行相似度聚类） 线上得分：0.2637
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

def disambiguate_by_cluster():
    res_dict = {}
    for author in validate_data:
        print(author)
        coauther_orgs = []
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            authors = paper['authors']
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]  
            abstract = paper["abstract"] if 'abstract' in paper else ''
            coauther_orgs.append(etl(' '.join(names + orgs) + ' '+ abstract))         
        tfidf = TfidfVectorizer().fit_transform(coauther_orgs)
        # sim_mertric = pairwise_distances(tfidf, metric='cosine')

        clf = DBSCAN(metric='cosine')
        s = clf.fit_predict(tfidf)
        #每个样本所属的簇
        for label, paper in zip(clf.labels_, papers):
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id'])
        res_dict[author] = list(paper_dict.values())
    json.dump(res_dict, open('result/disambiguate_by_cluster_result.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_cluster()


# 4. 半监督聚类（根据训练集构造训练pair对--->训练相似度模型---->利用相似度模型预测验证集相似度矩阵--->根据相似度矩阵进行聚类）
import math
from functools import reduce

# 获取标签
def get_label(paper_author_dict, id_i, id_j):
    if paper_author_dict[id_i] == paper_author_dict[id_j]:
        return 1
    return 0

# 1).构造训练集pairs
def gen_train_data(train_data, train_assign):
    sasn = []
    dasn = []
    for i, author in enumerate(train_data):
        print(author)  
        papers = [p_id for author_id in train_data[author] for p_id in train_data[author][author_id]]
        paper_author_dict = {p_id: author_id for author_id in train_data[author] for p_id in train_data[author][author_id]}
        print(len(papers))

        for i, j in zip(*np.triu_indices(len(papers), k=1)):
            p_i = papers[i]
            p_j = papers[j]

            # 标签
            label = get_label(paper_author_dict, p_i, p_i)
            if label == 0:
                dasn.append((p_i, p_i, 0))
            else:
                sasn.append((p_i, p_i, 1))

    print("len of sasn:", len(sasn))
    print("len of dasn:", len(dasn))

    category_size = len(sasn)
    all_pairs = map(lambda x: int(math.ceil(
        category_size / float(len(x)))) * x,[sasn, dasn])

    pairs = reduce(lambda x, y: x + random.sample(y, int(category_size)), all_pairs, [])
    return pairs

pairs = gen_train_data(train_data, train_pub_data)

# 2).提取特征训练相似度模型



# 3).预测验证集相似度矩阵



# 4).聚类消歧