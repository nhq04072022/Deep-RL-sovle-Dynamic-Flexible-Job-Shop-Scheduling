import numpy as np

'''
this module contains the job sequencing rules used in the experiment
sequencing agents may choose to follow one of following rules
or choose to use trained parameters for decision-making
'''

# 1️ Random: Chọn ngẫu nhiên một công việc trong hàng đợi
def random_sequencing(data):
    # Chọn ngẫu nhiên một chỉ số công việc từ danh sách
    job_position = np.random.randint(len(data[0]))
    return job_position


# 2️ Shortest Processing Time (SPT): Ưu tiên công việc có thời gian xử lý ngắn nhất
def SPT(data):
    # data[0] chứa thời gian xử lý từng công việc
    job_position = np.argmin(data[0])
    return job_position


# 3️ Longest Processing Time (LPT): Ưu tiên công việc có thời gian xử lý dài nhất
def LPT(data):
    job_position = np.argmax(data[0])
    return job_position


# 4️ Least Remaining Operations (LRO): Công việc còn ít công đoạn nhất
def LRO(data):
    # data[10] chứa số công đoạn còn lại
    job_position = np.argmax(data[10])
    return job_position


# 5️ Least Work Remaining (LWKR): Tổng công việc còn lại ít nhất
def LWKR(data):
    # Tổng thời gian còn lại = thời gian xử lý + thời gian chờ
    job_position = np.argmin(data[0] + data[1])
    return job_position


# 6️ LWKR + SPT: Kết hợp hai chỉ tiêu
def LWKRSPT(data):
    # Nhân đôi PT để ưu tiên SPT hơn
    job_position = np.argmin(data[0]*2 + data[1])
    return job_position


# 7️ LWKR + MOD (Modified Operational Due Date)
def LWKRMOD(data):
    due = data[2]  # hạn chót
    operational_finish = data[0] + data[3]  # thời điểm kết thúc sau công đoạn hiện tại
    MOD = np.max([due, operational_finish], axis=0)  # chỉ tiêu MOD
    job_position = np.argmin(data[0] + data[1] + MOD)
    return job_position


# 8️ Earliest Due Date (EDD): Ưu tiên công việc có deadline sớm nhất
def EDD(data):
    job_position = np.argmin(data[2])
    return job_position


# 9️ COVERT: Cost Over Time Rule
def COVERT(data):
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0, None)
    # Ưu tiên cao nếu cost thấp hoặc PT nhỏ
    priority = (1 - cost / (0.05 * average_pt)).clip(0, None) / data[0]
    job_position = priority.argmax()
    return job_position


# 10 Critical Ratio (CR): Tỷ lệ giữa thời gian còn lại và tổng thời gian yêu cầu
def CR(data):
    CR = data[5] / (data[0] + data[1])
    job_position = CR.argmin()
    return job_position


# 11 CR kết hợp với SPT
def CRSPT(data):
    CRSPT = data[5] / (data[0] + data[1]) + data[0]
    job_position = CRSPT.argmin()
    return job_position


# 12 Minimum Slack: Chọn công việc có độ trễ nhỏ nhất
def MS(data):
    slack = data[6]
    job_position = slack.argmin()
    return job_position


# 13 Modified Due Date (MDD)
def MDD(data):
    finish = data[1] + data[3]
    MDD = np.max([data[2], finish], axis=0)
    job_position = MDD.argmin()
    return job_position


# 14 Montagne Rule: Dựa trên tỷ số giữa deadline và tổng PT
def MON(data):
    due_over_pt = np.array(data[2]) / np.sum(data[0])
    priority = due_over_pt / np.array(data[0])
    job_position = priority.argmax()
    return job_position


# 15 MOD: Modified Operational Due Date
def MOD(data):
    operational_finish = data[0] + data[3]
    MOD = np.max([data[2], operational_finish], axis=0)
    job_position = MOD.argmin()
    return job_position


# 16 NPT: Next Processing Time – thời gian xử lý tiếp theo
def NPT(data):
    job_position = np.argmin(data[9])
    return job_position


# 17 Apparent Tardiness Cost (ATC): heuristic phổ biến trong FJSP
def ATC(data):
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0, None)
    priority = np.exp(-cost / (0.05 * average_pt)) / data[0]
    job_position = priority.argmax()
    return job_position


# 18 AVPRO: Average Processing Time per Operation
def AVPRO(data):
    AVPRO = (data[0] + data[1]) / (data[10] + 1)
    job_position = AVPRO.argmin()
    return job_position


def SRMWK(data): # slack per remaining work, identical to CR
    SRMWK = data[6] / (data[0] + data[1])
    job_position = SRMWK.argmin()
    return job_position

def SRMWKSPT(data): # slack per remaining work + SPT, identical to CR+SPT
    SRMWKSPT = data[6] / (data[0] + data[1]) + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position

# 🔁 WINQ: Work in Next Queue – công việc ở hàng đợi sau
def WINQ(data):
    job_position = data[7].argmin()
    return job_position


def PTWINQ(data): # PT + WINQ
    sum = data[0] + data[7]
    job_position = sum.argmin()
    return job_position

def PTWINQS(data): # PT + WINQ + Slack
    sum = data[0] + data[6] + data[7]
    job_position = sum.argmin()
    return job_position

def DPTWINQNPT(data): # 2PT + WINQ + NPT
    sum = data[0]*2 + data[7] + data[9]
    job_position = sum.argmin()
    return job_position

def DPTLWKR(data): # 2PT + LWKR
    sum = data[0]*2 + data[1]
    job_position = sum.argmin()
    return job_position

def DPTLWKRS(data): # 2PT + LWKR + slack
    sum = data[0]*2 + data[1] + data[6]
    job_position = sum.argmin()
    return job_position

def FIFO(dummy): # first in, first out, data is not needed
    job_position = 0
    return job_position

def GP_S1(data): # genetic programming rule 1
    eps = 1e-8  # giá trị rất nhỏ để tránh chia cho 0

    sec1 = data[0] + data[1]

    safe_data0 = np.where(np.abs(data[0]) < eps, eps, data[0])
    safe_denom1 = np.where(np.abs(data[7] - data[1]) < eps, eps, data[7] - data[1])

    sec2 = (data[7] * 2 - 1) / safe_data0
    sec3 = (data[7] + data[1] + (data[0] + data[1]) / safe_denom1) / safe_data0

    sum = sec1 - sec2 - sec3
    job_position = sum.argmin()
    return job_position


def GP_S2(data): # genetic programming rule 2
    NIQ = len(data[0]) # số công việc trong hàng đợi
    sec1 = NIQ * (data[0]-1)
    sec2 = data[0] + data[1] * np.max([data[0],data[7]],axis=0)
    sec3 = np.max([data[7],NIQ+data[7]],axis=0)
    sec4 = (data[8]+1+np.max([data[1],np.ones_like(data[1])*(NIQ-1)],axis=0)) * np.max([data[7],data[1]],axis=0)
    sum = sec1 * sec2 + sec3 * sec4
    job_position = sum.argmin()
    return job_position

# def GP_S3(data): # genetic programming rule 1
#     sec1 = data[0] + data[1]
#     sec2 = (data[7]*2-1) / data[0]
#     sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
#     sum = sec1-sec2-sec3
#     job_position = sum.argmin()
#     return job_position
