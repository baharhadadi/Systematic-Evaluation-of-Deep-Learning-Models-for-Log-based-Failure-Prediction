import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import numpy as np
from sklearn import datasets, tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# read the results of BERT
df1tr_b = pd.read_csv("Results/transformer_BERT_M1.csv").dropna()
df1ls_b = pd.read_csv("Results/LSTM_BERT_M1.csv").dropna()
df1cn_b = pd.read_csv("Results/CNN_BERT_M1.csv").dropna()
df1bi_b  = pd.read_csv("Results/BiLSTM_BERT_M1.csv").dropna()
df2tr_b = pd.read_csv("Results/transformer_BERT_M2.csv").dropna()
df2ls_b = pd.read_csv("Results/LSTM_BERT_M2.csv").dropna()
df2cn_b = pd.read_csv("Results/CNN_BERT_M2.csv").dropna()
df2bi_b  = pd.read_csv("Results/BiLSTM_BERT_M2.csv").dropna()
df3tr_b = pd.read_csv("Results/transformer_BERT_M3.csv").dropna()
df3ls_b = pd.read_csv("Results/LSTM_BERT_M3.csv").dropna()
df3cn_b = pd.read_csv("Results/CNN_BERT_M3.csv").dropna()
df3bi_b  = pd.read_csv("Results/BiLSTM_BERT_M3.csv").dropna()

# read the results of Logkey2vec
df1tr_l = pd.read_csv("Results/transformer_Logkey2vec_M1.csv").dropna()
df1ls_l = pd.read_csv("Results/LSTM_Logkey2vec_M1.csv").dropna()
df1cn_l = pd.read_csv("Results/CNN_Logkey2vec_M1.csv").dropna()
df1bi_l = pd.read_csv("Results/BiLSTM_Logkey2vec_M1.csv").dropna()
df2tr_l = pd.read_csv("Results/transformer_Logkey2vec_M2.csv").dropna()
df2ls_l = pd.read_csv("Results/LSTM_Logkey2vec_M2.csv").dropna()
df2cn_l = pd.read_csv("Results/CNN_Logkey2vec_M2.csv").dropna()
df2bi_l  = pd.read_csv("Results/BiLSTM_Logkey2vec_M2.csv").dropna()
df3tr_l = pd.read_csv("Results/transformer_Logkey2vec_M3.csv").dropna()
df3ls_l = pd.read_csv("Results/LSTM_Logkey2vec_M3.csv").dropna()
df3cn_l = pd.read_csv("Results/CNN_Logkey2vec_M3.csv").dropna()
df3bi_l  = pd.read_csv("Results/BiLSTM_Logkey2vec_M3.csv").dropna()

# read the results of BERT
df1tr_f = pd.read_csv("Results/transformer_F+T_M1.csv").dropna()
df1ls_f = pd.read_csv("Results/LSTM_F+T_M1.csv").dropna()
df1cn_f = pd.read_csv("Results/CNN_F+T_M1.csv").dropna()
df1bi_f  = pd.read_csv("Results/BiLSTM_F+T_M1.csv").dropna()
df2tr_f = pd.read_csv("Results/transformer_F+T_M2.csv").dropna()
df2ls_f = pd.read_csv("Results/LSTM_F+T_M2.csv").dropna()
df2cn_f = pd.read_csv("Results/CNN_F+T_M2.csv").dropna()
df2bi_f  = pd.read_csv("Results/BiLSTM_F+T_M2.csv").dropna()
df3tr_f = pd.read_csv("Results/transformer_F+T_M3.csv").dropna()
df3ls_f = pd.read_csv("Results/LSTM_F+T_M3.csv").dropna()
df3cn_f = pd.read_csv("Results/CNN_F+T_M3.csv").dropna()
df3bi_f  = pd.read_csv("Results/BiLSTM_F+T_M3.csv").dropna()

rf_1 = pd.read_csv("Results/RF_M1.csv").dropna()
rf_2 = pd.read_csv("Results/RF_M2.csv").dropna()
rf_3  = pd.read_csv("Results/RF_M3.csv").dropna()

# concat the results of all data collections for each combination
df1tr = pd.concat([df1tr_l, df1tr_b, df1tr_f])
df2tr = pd.concat([df2tr_l, df2tr_b, df2tr_f])
df3tr = pd.concat([df3tr_l, df3tr_b, df3tr_f])

df1ls = pd.concat([df1ls_l, df1ls_b, df1ls_f])
df2ls = pd.concat([df2ls_l, df2ls_b, df2ls_f])
df3ls = pd.concat([df3ls_l, df3ls_b, df3ls_f])

df1cn = pd.concat([df1cn_l, df1cn_b, df1cn_f])
df2cn = pd.concat([df2cn_l, df2cn_b, df2cn_f])
df3cn = pd.concat([df3cn_l, df3cn_b, df3cn_f])

df1bi = pd.concat([df1bi_l, df1bi_b, df1bi_f])
df2bi = pd.concat([df2bi_l, df2bi_b, df2bi_f])
df3bi = pd.concat([df3bi_l, df3bi_b, df3bi_f])

dftr_b = pd.concat([df1tr_b, df2tr_b, df3tr_b])
dftr_l = pd.concat([df1tr_l, df2tr_l, df3tr_l])
dftr_f = pd.concat([df1tr_f, df2tr_f, df3tr_f])

dfcn_b = pd.concat([df1cn_b, df2cn_b, df3cn_b])
dfcn_l = pd.concat([df1cn_l, df2cn_l, df3cn_l])
dfcn_f = pd.concat([df1cn_f, df2cn_f, df3cn_f])

dfbi_b = pd.concat([df1bi_b, df2bi_b, df3bi_b])
dfbi_l = pd.concat([df1bi_l, df2bi_l, df3bi_l])
dfbi_f = pd.concat([df1bi_f, df2bi_f, df3bi_f])

dfls_b = pd.concat([df1ls_b, df2ls_b, df3ls_b])
dfls_l = pd.concat([df1ls_l, df2ls_l, df3ls_l])
dfls_f = pd.concat([df1ls_f, df2ls_f, df3ls_f])

df1_b = pd.concat([df1ls_b, df1tr_b, df1cn_b, df1bi_b])
df2_b = pd.concat([df2ls_b, df2tr_b, df2cn_b, df2bi_b])
df3_b = pd.concat([df3ls_b, df3tr_b, df3cn_b, df3bi_b])

df1_f = pd.concat([df1ls_f, df1tr_f, df1cn_f, df1bi_f])
df2_f = pd.concat([df2ls_f, df2tr_f, df2cn_f, df2bi_f])
df3_f = pd.concat([df3ls_f, df3tr_f, df3cn_f, df3bi_f])

df1_l = pd.concat([df1ls_l, df1tr_l, df1cn_l, df1bi_l])
df2_l = pd.concat([df2ls_l, df2tr_l, df2cn_l, df2bi_l])
df3_l = pd.concat([df3ls_l, df3tr_l, df3cn_l, df3bi_l])

df_b = pd.concat([df1_b, df2_b, df3_b])
df_f = pd.concat([df1_f, df2_f, df3_f])
df_l = pd.concat([df1_l, df2_l, df3_l])

dfcn = pd.concat([df1cn, df2cn, df3cn])
dfbi = pd.concat([df1bi, df2bi, df3bi])
dfls = pd.concat([df1ls, df2ls, df3ls])

###################################################
# RQ1

bars1 = [df1tr["F_F1"],df1ls["F_F1"],df1cn["F_F1"],df1bi["F_F1"]]

bars2 = [df2tr["F_F1"],df2ls["F_F1"],df2cn["F_F1"],df2bi["F_F1"]]

bars3 =[df3tr["F_F1"],df3ls["F_F1"],df3cn["F_F1"],df3bi["F_F1"]]

ticks = ["transformer-based","LSTM-based","CNN-based","Bi-LSTM+attention"]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#ed7d34', label='$\mathcal{M}_1$',linestyle='solid')
plt.plot([], c='#5aa4d8', label='$\mathcal{M}_2$',linestyle='solid')
plt.plot([], c="#3a6186", label='$\mathcal{M}_3$',linestyle='solid')
plt.legend()

#plt.text(-1.3, bars1[0].mean()+0.02,str(round(bars1[0].mean(),2)),fontsize=10)
plt.xticks([-0.9+0.4 , 1.2+0.5, 3+0.5, 5.2+0.5], ticks, rotation= 20)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Failure Predictor Models', fontweight='bold')
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.title("Box Plot of F1 Score Results")
plt.savefig('rq1-boxplot.pdf', bbox_inches="tight")

####################################################
# RQ2

# plotting the boxplot of all DL encoders for each embedding strategy
bars1 = [df1_l["F_F1"], df1_f["F_F1"], df1_b["F_F1"]]

bars2 = [df2_l["F_F1"], df2_f["F_F1"], df2_b["F_F1"]]

bars3 =[df3_l["F_F1"], df3_f["F_F1"], df3_b["F_F1"]]

ticks = ["Logkey2vec", "F+T", "BERT"]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.8, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#ed7d34', label='$\mathcal{M}_1$',linestyle='solid')
plt.plot([], c='#5aa4d8', label='$\mathcal{M}_2$',linestyle='solid')
plt.plot([], c="#3a6186", label='$\mathcal{M}_3$',linestyle='solid')
plt.legend(loc= "upper right")

#plt.text(-1.3, bars1[0].mean()+0.02,str(round(bars1[0].mean(),2)),fontsize=10)
plt.xticks([0.2 , 2.2, 4.2], ticks, rotation= 20)
plt.xlim(-1, len(ticks)*2+1)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Log Sequence Embedding Strategy', fontweight='bold')
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.title("Box Plot of F1 Score Results")
plt.savefig('rq2-boxplot-combined.pdf', bbox_inches="tight")
#####################

# plotting the boxplot of CNN
bars1 = [df1cn_l["F_F1"], df1cn_f["F_F1"], df1cn_b["F_F1"]]

bars2 = [df2cn_l["F_F1"], df2cn_f["F_F1"], df2cn_b["F_F1"]]

bars3 =[df3cn_l["F_F1"], df3cn_f["F_F1"], df3cn_b["F_F1"]]

ticks = ["Logkey2vec", "F+T", "BERT"]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.8, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#ed7d34', label='$\mathcal{M}_1$',linestyle='solid')
plt.plot([], c='#5aa4d8', label='$\mathcal{M}_2$',linestyle='solid')
plt.plot([], c="#3a6186", label='$\mathcal{M}_3$',linestyle='solid')
plt.legend(loc= "upper right")

#plt.text(-1.3, bars1[0].mean()+0.02,str(round(bars1[0].mean(),2)),fontsize=10)
plt.xticks([0.2 , 2.2, 4.2], ticks, rotation= 20)
plt.xlim(-1, len(ticks)*2+1)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Log Sequence Embedding Strategy', fontweight='bold')
plt.ylim(0.32, 1.05)
plt.tight_layout()
plt.title("CNN-based")
plt.savefig('rq2-boxplot-cnn.pdf', bbox_inches="tight")

#######################################################
# RQ3

# box plot of best configuration (CNN+Logkey2vec) and RF

bars1 = [df1cn_l["F_F1"],rf_1["F_F1"]]

bars2 = [df2cn_l["F_F1"],rf_2["F_F1"]]

bars3 =[df3cn_l["F_F1"],rf_3["F_F1"]]

ticks = ["CNN + L","RF"]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#ed7d34', label='$\mathcal{M}_1$',linestyle='solid')
plt.plot([], c='#5aa4d8', label='$\mathcal{M}_2$',linestyle='solid')
plt.plot([], c="#3a6186", label='$\mathcal{M}_3$',linestyle='solid')
plt.legend()

#plt.text(-1.3, bars1[0].mean()+0.02,str(round(bars1[0].mean(),2)),fontsize=10)
plt.xticks([-0.9+0.4 , 1.2+0.5], ticks, rotation= 20)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Failure Predictor Models', fontweight='bold')
plt.ylim(0.5, 1.05)
plt.tight_layout()
plt.title("Box Plot of F1 Score Results")
plt.savefig('rq3-cnn-rf.pdf', bbox_inches="tight")

#######################################################
# RQ4

# boxplot of dataset size for CNN with Logkey2vec
ticks = ["200","500","1000","5000","10000","50000"]
levels = [200,500,1000,5000,10000,50000]


# set heights of bars
bars1 = [df1cn_l.loc[df1cn_l["DS"] == l]["F_F1"] for l in levels]
bars2 = [df2cn_l.loc[df2cn_l["DS"] == l]["F_F1"] for l in levels]
bars3 = [df3cn_l.loc[df3cn_l["DS"] == l]["F_F1"] for l in levels]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#ed7d34', label='$\mathcal{M}_1$',linestyle='solid')
plt.plot([], c='#5aa4d8', label='$\mathcal{M}_2$',linestyle='solid')
plt.plot([], c="#3a6186", label='$\mathcal{M}_3$',linestyle='solid')
plt.legend()

plt.xticks([-0.8+0.5 , 1.2+0.5, 3.2+0.5, 5.2+0.5, 7.2+0.5, 9.2+0.5], ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Data Set Size Levels', fontweight='bold')
plt.ylim(0.6, 1.02)
plt.tight_layout()
#plt.title("(a)")
plt.savefig('rq4-cnn-l-a.pdf', bbox_inches="tight")
#####################

# boxplot of maximum log sequence length 
levels = [20,50,100,500,1000]
ticks = list(map(str,levels))

# set heights of bars
bars1 = [df1cn_l.loc[df1cn_l["MLSL"] == l]["F_F1"] for l in levels]
bars2 = [df2cn_l.loc[df2cn_l["MLSL"] == l]["F_F1"] for l in levels]
bars3 = [df3cn_l.loc[df3cn_l["MLSL"] == l]["F_F1"] for l in levels]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

plt.xticks([-0.8+0.5 , 1.2+0.5, 3.2+0.5, 5.2+0.5, 7.2+0.5], ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Maximum Sequence Length Levels', fontweight='bold')
plt.ylim(0.6, 1.02)
plt.tight_layout()
#plt.title("(b)")
plt.savefig('rq4-cnn-l-b.pdf', bbox_inches="tight")
#####################

# boxplot of failure precentage
levels = [5,10,20,30,40,50]
ticks = list(map(str,levels))

# set heights of bars
bars1 = [df1cn_l.loc[df1cn_l["Fperc"] == l]["F_F1"] for l in levels]
bars2 = [df2cn_l.loc[df2cn_l["Fperc"] == l]["F_F1"] for l in levels]
bars3 = [df3cn_l.loc[df3cn_l["Fperc"] == l]["F_F1"] for l in levels]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")


plt.xticks([-0.8+0.5 , 1.2+0.5, 3.2+0.5, 5.2+0.5, 7.2+0.5, 9.2+0.5], ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Failure Percentage Levels', fontweight='bold')
plt.ylim(0.6, 1.02)
plt.tight_layout()
#plt.title("(c)")
plt.savefig('rq4-cnn-l-c.pdf', bbox_inches="tight")
#####################

# boxplot of failure pattern types
levels = [1, 2]
ticks = list(map(str,levels))

# set heights of bars
bars1 = [df1cn_l.loc[df1cn_l["FP"] == l]["F_F1"] for l in levels]
bars2 = [df2cn_l.loc[df2cn_l["FP"] == l]["F_F1"] for l in levels]
bars3 = [df3cn_l.loc[df3cn_l["FP"] == l]["F_F1"] for l in levels]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")


plt.xticks([-0.8+0.5 , 1.2+0.5], ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Failure Pattern Types', fontweight='bold')
plt.ylim(0.6, 1.02)
plt.tight_layout()
#plt.title("(d)")
plt.savefig('rq4-cnn-l-d.pdf', bbox_inches="tight")
#####################

# boxplot of maximum log sequence length for BiLSTM+BERT
dfcn = pd.concat([df1cn["F_F1"], df2cn["F_F1"], df3cn["F_F1"]])
levels = [20,50,100,500,1000]
ticks = list(map(str,levels))

# set heights of bars
bars1 = [df1bi_b.loc[df1bi_b["MLSL"] == l]["F_F1"] for l in levels]
bars2 = [df2bi_b.loc[df2bi_b["MLSL"] == l]["F_F1"] for l in levels]
bars3 = [df3bi_b.loc[df3bi_b["MLSL"] == l]["F_F1"] for l in levels]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

box1 = plt.boxplot(bars1, positions=np.array(range(len(bars1)))*2.0-1, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#ed7d34',
                      "markersize":"3"})
box2 = plt.boxplot(bars2, positions=np.array(range(len(bars2)))*2.0-0.4, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":'#5aa4d8',
                      "markersize":"3"})
box3 = plt.boxplot(bars3, positions=np.array(range(len(bars3)))*2.0+0.2, sym='', widths=0.4, showmeans = True,meanprops={
                       "markerfacecolor":"white",
                       "markeredgecolor":"#3a6186",
                      "markersize":"3"})
set_box_color(box1, '#ed7d34') # colors are from http://colorbrewer2.org/
set_box_color(box2, '#5aa4d8')
set_box_color(box3, "#3a6186")

plt.xticks([-0.8+0.5 , 1.2+0.5, 3.2+0.5, 5.2+0.5, 7.2+0.5], ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylabel('F1 Score for Failure Class', fontweight='bold')
plt.xlabel('Maximum Sequence Length Levels', fontweight='bold')
plt.ylim(-0.02, 1.02)
plt.tight_layout()
#plt.title("(b)")
plt.savefig('rq4-bilstm-b-b.pdf', bbox_inches="tight")
#####################

# decision tree of top three configurations
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Prepare the data
dfs = {}
x_values = ['cn', 'bi', 'ls', 'tr']
y_values = ['l', 'b', 'f']

for x in x_values:
    for y in y_values:
        df_name = 'df' + x + '_' + y
        dfs[df_name] = globals()[df_name]
for df_name in dfs:
    dfs[df_name] = dfs[df_name].reset_index(drop=True)

# Create an empty dataframe to store the results
result_df_3 = pd.DataFrame(index=range(1080), columns=['Max_F1_DFs'])

for i in range(1080):
    max_val = float('-inf')  # initialize with negative infinity
    max_dfs = []

    # Step 2: Loop through each dataframe and compare 'F_F1' values
    for df_name, df in dfs.items():
      if df_name in ["dfcn_l", "dfcn_b", "dfbi_b"]:
        if df.at[i, 'F_F1'] > max_val:
            max_val = df.at[i, 'F_F1']
            max_dfs = [df_name]
        elif df.at[i, 'F_F1'] == max_val:
            max_dfs.append(df_name)

    # Step 3: Store the dataframe name(s) in the result dataframe
    result_df_3.at[i, 'Max_F1_DFs'] = ', '.join(max_dfs)

# Create an empty dataframe to store the results
result_df = pd.DataFrame(index=range(1080), columns=['Max_F1_DFs_0.01'])

threshold = 0.01

for i in range(1080):
    max_val = float('-inf')  # initialize with negative infinity
    max_dfs = []

    # Step 2: Loop through each dataframe and compare 'F_F1' values
    for df_name, df in dfs.items():
        diff = df.at[i, 'F_F1'] - max_val

        # If the value in the current dataframe is significantly greater than max_val
        if diff > threshold:
            max_val = df.at[i, 'F_F1']
            max_dfs = [df_name]
        # If the difference between the value in the current dataframe and max_val is within the threshold
        elif abs(diff) < threshold:
            max_dfs.append(df_name)

    # Step 3: Store the dataframe name(s) in the result dataframe
    result_df.at[i, 'Max_F1_DFs_0.01'] = ', '.join(max_dfs)

scaling_factor =1 # this can be tuned

# Raw weights (full equalization)
raw_weight_CNN_L = 1
raw_weight_BiLSTM_B = 573 / 74
raw_weight_CNN_B = 573 / 76

# Scaled weights
weight_CNN_L = 1
weight_BiLSTM_B = 1 + scaling_factor * (raw_weight_BiLSTM_B - 1)
weight_CNN_B = 1 + scaling_factor * (raw_weight_CNN_B - 1)

# Resulting class weights
class_weights = {
    'CNN+L': weight_CNN_L,
    'BiLSTM+B': weight_BiLSTM_B,
    'CNN+B': weight_CNN_B
}


# Create a column in result_df for the labels based on your conditions
def assign_label(row):
    if 'dfcn_l' in row['Max_F1_DFs_0.01']:
        return 'CNN+L'
    elif 'dfbi_b' in row['Max_F1_DFs_0.01'] :
        return 'CNN+B'
    else:
        return 'BiLSTM+B'

result_df_3['label'] = result_df.apply(assign_label, axis=1)

# One-hot encode the "FP" column
X = pd.get_dummies(dfcn_l[['FP', 'MLSL', 'DS', 'Fperc']], columns=['FP'], drop_first=True)  # drop_first removes multicollinearity

# Extract only relevant columns. Since get_dummies creates new columns, we need to adjust this
feature_columns = [col for col in X.columns if col not in ['Max_F1_DFs_0.01', 'label']]
X = X[feature_columns]
X = dfcn_l[[ 'MLSL', 'DS', 'Fperc']]
y = result_df_3['label']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=92)

# Train a decision tree without pruning and compute the impurities for each node
path = DecisionTreeClassifier(random_state=92).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train a decision tree for each ccp_alpha and store the accuracy for the validation set
accuracies = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha,class_weight=class_weights)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    accuracies.append(accuracy_score(y_valid, y_pred))

# Get the optimal ccp_alpha value with the maximum accuracy
optimal_ccp_alpha = ccp_alphas[accuracies.index(max(accuracies))]

clf_pruned = DecisionTreeClassifier(
    random_state=0,
    ccp_alpha=optimal_ccp_alpha*9 ,  # Increase the ccp_alpha for more pruning
    max_depth=3,                         # Limiting the depth
    min_samples_split=109,                # Minimum samples for a node to split
    min_samples_leaf=109,                  # Minimum samples for a leaf
    class_weight=class_weights
)
clf_pruned.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(clf_pruned, filled=True, feature_names=X.columns, class_names=clf_pruned.classes_, rounded=True)
plt.savefig('rq4-decisiontree.pdf', bbox_inches="tight")

#####################

# regression tree of CNN with Logkey2vec

# defining the regressor
regressor = DecisionTreeRegressor(random_state = 0 , min_samples_leaf = 2, min_samples_split = 3, max_depth = 5) 

# keeping the not needed columns
df = dfcn_l.drop(['N_R', 'N_P', 'N_F1', 'F_R', 'F_P'], axis=1)

# spliting the data to triana and test
train, test = train_test_split(df, test_size=1/3, random_state = 42)

# process the data 
# so that the regressor can regoznize which variable is categorical and which one is continues
y = train['F_F1']
x = train.drop('F_F1', axis=1)

# fit the regressor
regressor.fit(x, y)

# prone the regression tree to prevent from overfitting
path = regressor.cost_complexity_pruning_path(x, y)
param_grid = {"ccp_alpha": path.ccp_alphas}
grid = GridSearchCV(estimator=regressor, param_grid=param_grid)
grid.fit(x, y)
regressor = grid.best_estimator_

# plot the proned tree
plt.figure()
tree.plot_tree(regressor, feature_names=["FPtype", "MLSL","DS","Fperc"])  
plt.savefig('rq4-cnn-l-regressiontree.pdf')

#########################

# regression tree of CNN with BERT

# defining the regressor
regressor = DecisionTreeRegressor(random_state = 0 , min_samples_leaf = 2, min_samples_split = 3, max_depth = 5) 

# keeping the not needed columns
df = dfcn_b.drop(['N_R', 'N_P', 'N_F1', 'F_R', 'F_P'], axis=1)

# spliting the data to triana and test
train, test = train_test_split(df, test_size=.33, random_state = 42)

# process the data 
# so that the regressor can regoznize which variable is categorical and which one is continues
y = train['F_F1']
x = train.drop('F_F1', axis=1)

# fit the regressor
regressor.fit(x, y)

# prone the regression tree to prevent from overfitting
path = regressor.cost_complexity_pruning_path(x, y)
param_grid = {"ccp_alpha": path.ccp_alphas}
grid = GridSearchCV(estimator=regressor, param_grid=param_grid)
Y = []
for i in y:
  if i>0.7:
    Y.append(1)
  else:
    Y.append(0)
Y = pd.DataFrame(Y)
grid.fit(x, Y)
regressor = grid.best_estimator_

# plot the proned tree
plt.figure()
tree.plot_tree(regressor, feature_names=["FPtype", "MLSL","DS","Fperc"])  
plt.savefig('rq4-cnn-b-regressiontree.pdf')

############################

# regression tree of BiSLTM with BERT

# defining the regressor
regressor = DecisionTreeRegressor(random_state = 0 , min_samples_leaf = 2, min_samples_split = 3, max_depth = 5) 

# keeping the not needed columns
df = dfbi_b.drop(['N_R', 'N_P', 'N_F1', 'F_R', 'F_P'], axis=1)

# spliting the data to triana and test
train, test = train_test_split(df, test_size=.33, random_state = 7)

# process the data 
# so that the regressor can regoznize which variable is categorical and which one is continues
y = train['F_F1']
x = train.drop('F_F1', axis=1)

# fit the regressor
regressor.fit(x, y)

# prone the regression tree to prevent from overfitting
path = regressor.cost_complexity_pruning_path(x, y)
param_grid = {"ccp_alpha": path.ccp_alphas*4}
grid = GridSearchCV(estimator=regressor, param_grid=param_grid)
Y = []
for i in y:
  if i>0.7:
    Y.append(1)
  else:
    Y.append(0)
Y = pd.DataFrame(Y)
grid.fit(x, Y)
regressor = grid.best_estimator_

# plot the proned tree
plt.figure()
tree.plot_tree(regressor, feature_names=["FPtype", "MLSL","DS","Fperc"])  
plt.savefig('rq4-bilstm-b-regressiontree.pdf')
