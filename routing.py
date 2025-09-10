import numpy as np

'''
Mô-đun này chứa các **thuật toán định tuyến (routing rules)** được sử dụng làm benchmark để so sánh.
Routing agent có thể:
- chọn một trong các luật này (heuristic), hoặc
- sử dụng tham số đã huấn luyện (deep RL).
'''

# 1️ Rule "random_routing": chọn máy hoàn toàn ngẫu nhiên
def random_routing(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.random.randint(len(job_pt))  # Chọn ngẫu nhiên trong số lượng máy
    return machine_idx

# 2️ Rule "TT" - Shortest Total Waiting Time (tổng thời gian chờ ngắn nhất)
def TT(idx, data, job_pt, job_slack, wc_idx, *args):
    # data: ma trận đặc trưng [tải, thời gian khả dụng, độ dài hàng đợi, ...]
    rank = np.argmin(data, axis=0)  # Lấy chỉ số min theo cột (axis=0)
    machine_idx = rank[0]  # Cột 0 biểu thị tổng công việc đang chờ
    return machine_idx

# 3️ Rule "ET" - Minimum Execution Time (thời gian xử lý ngắn nhất)
def ET(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.argmin(job_pt)  # chọn máy có thời gian xử lý job này ngắn nhất
    return machine_idx

# 4️ Rule "EA" - Earliest Available (thời gian rảnh sớm nhất)
def EA(idx, data, job_pt, job_slack, wc_idx, *args):
    rank = np.argmin(data, axis=0)  # chọn cột có giá trị nhỏ nhất
    machine_idx = rank[1]  # cột 1 là thời gian rảnh sớm nhất
    return machine_idx

# 5️⃣ Rule "SQ" - Shortest Queue (độ dài hàng đợi ngắn nhất)
def SQ(idx, data, job_pt, job_slack, wc_idx, *args):
    rank = np.argmin(data, axis=0)
    machine_idx = rank[2]  # cột 2: độ dài hàng đợi
    return machine_idx

# 6️ Rule "CT" - Earliest Completion Time (thời gian hoàn thành sớm nhất)
def CT(idx, data, job_pt, job_slack, wc_idx, *args):
    # data[:,1] là thời gian rảnh của máy → thời điểm bắt đầu xử lý job này
    completion_time = np.array(data)[:,1].clip(0) + np.array(job_pt)
    machine_idx = completion_time.argmin()  # chọn máy hoàn thành job sớm nhất
    return machine_idx

# 7️ Rule "UT" - Lowest Utilization Rate (tỷ lệ sử dụng thấp nhất)
def UT(idx, data, job_pt, job_slack, wc_idx, *args):
    rank = np.argmin(data, axis=0)
    machine_idx = rank[3]  # cột 3 là tỷ lệ sử dụng
    return machine_idx
def GP_R1(idx, data, job_pt, job_slack, wc_idx, *args):
    """
    Genetic Programming-based routing rule (phiên bản 1).
    Sử dụng tổ hợp các phép toán được GP tiến hóa.
    """
    eps = 1e-6  # để tránh chia 0

    data = np.transpose(data)  # chuyển data về dạng (3, num_machines)
    data0 = data[0]  # tải hoặc công việc trong máy
    data1 = np.where(data[1] == 0, eps, data[1])  # thời gian sẵn sàng
    data2 = data[2]  # độ dài hàng đợi

    # === sec1: tổ hợp phi tuyến phức tạp giữa data2, job_pt và data1
    term1 = data2 * job_pt / data1
    term2 = job_pt * data0**2
    max_term = np.maximum(term1, term2)
    sec1 = np.clip(2 * data2 * max_term, -1e6, 1e6)

    # === sec2: công thức khác giữa độ dài hàng đợi và thời gian khả dụng
    sec2 = data2 * job_pt - data1

    # Tổng các thành phần
    sum_score = sec1 + sec2

    machine_idx = np.argmin(sum_score)  # chọn máy có điểm thấp nhất
    return machine_idx
def GP_R2(idx, data, job_pt, job_slack, wc_idx, *args):
    data = np.transpose(data)
    eps = 1e-8

    # Các phần tử cần thiết
    data0 = data[0]
    data1 = data[1]
    data2 = data[2]

    # === sec1: gồm 2 biểu thức tách biệt
    sec1 = data2 * data2, (data2 + job_pt) * data2

    # === sec2: tránh chia 0
    denom_sec2 = np.where(np.abs(data1 * args[0] - 1) < eps, eps, data1 * args[0] - 1)
    sec2 = np.minimum(data1, args[0] / denom_sec2)

    # === sec3: negative term (trừ để giảm score)
    sec3 = -data2 * args[0]

    # === sec4: tổ hợp logic phức tạp
    denom_sec4 = np.where(np.abs(args[0]) < eps, eps, args[0])
    sec4_inner = np.minimum(data1, job_pt) / denom_sec4
    sec4 = data2 * job_pt * np.maximum(data0, sec4_inner)

    # === sec5: hàm phức tạp dùng max
    min_sec5 = np.minimum(data2, np.ones_like(data2) * args[1])
    sec5 = np.maximum.reduce([
        data2 * data2,
        np.ones_like(data2) * (args[1] - args[0] - 1),
        (data2 + job_pt) * min_sec5
    ])
    sec5 = np.where(np.abs(sec5) < eps, eps, sec5)  # tránh chia 0

    # === Tổng hợp
    sum_expr = sec3 + sec4 / sec5
    sum_val = np.maximum.reduce([
        sec1[0] - sec2 * sum_expr,
        sec1[1] - sec2 * sum_expr
    ])

    machine_idx = sum_val.argmin()
    return machine_idx
