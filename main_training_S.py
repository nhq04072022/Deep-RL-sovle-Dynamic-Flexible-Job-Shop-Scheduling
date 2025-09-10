# === Nhập các thư viện mô phỏng và học máy ===
import simpy                              # Thư viện mô phỏng mô hình rời rạc (DES)
import matplotlib.pyplot as plt           # Vẽ đồ thị
import matplotlib.animation as animation  # Tạo hoạt hình
import torch                              # Học sâu với PyTorch
import numpy as np                        # Xử lý ma trận, số liệu
from tabulate import tabulate             # Hiển thị bảng đẹp hơn

# === Nhập các module dùng cho mô hình hóa máy, trung tâm làm việc, luật xếp việc và định tuyến ===
import agent_machine                      # Lớp mô phỏng một máy
import agent_workcenter                   # Lớp mô phỏng một workcenter (trung tâm làm việc)
import brain_machine_S                    # DRL brain cho việc xếp thứ tự công việc (sequencing)
import job_creation                       # Module sinh công việc (job)
import breakdown_creation                 # Tạo sự cố
import heterogeneity_creation             # Tạo sự không đồng nhất giữa các máy

"""
MODULE DÙNG CHO HUẤN LUYỆN SẮP XẾP JOB TRÊN MÁY (SEQUENCING AGENT)
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        ''' BƯỚC 1: Tạo môi trường mô phỏng và thông số '''
        self.env = env                  # Môi trường SimPy
        self.span = span               # Tổng thời gian mô phỏng
        self.m_no = m_no               # Số lượng máy
        self.m_list = []               # Danh sách các máy
        self.wc_no = wc_no             # Số lượng workcenter
        self.wc_list = []              # Danh sách các workcenter
        m_per_wc = int(m_no / wc_no)   # Máy mỗi workcenter

        ''' BƯỚC 2.1: Tạo các máy '''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i)
            exec(expr1)               # Tạo đối tượng máy m_i
            expr2 = '''self.m_list.append(self.m_{})'''.format(i)
            exec(expr2)               # Thêm máy vào danh sách

        ''' BƯỚC 2.2: Tạo các workcenter, mỗi workcenter gồm nhiều máy '''
        cum_m_idx = 0
        for i in range(wc_no):
            # Lấy danh sách máy con theo chỉ số
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i)
            exec(expr1)               # Khởi tạo workcenter với các máy
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i)
            exec(expr2)
            cum_m_idx += m_per_wc

        ''' BƯỚC 3: Tạo bộ sinh công việc (job) '''
        self.job_creator = job_creation.creation(
            self.env,
            self.span,
            self.m_list,
            self.wc_list,
            [5, 26],         # Thời gian xử lý job ngẫu nhiên từ 5 đến 26
            3,               # Độ gấp rút của job (due date tightness)
            0.9,             # Tỷ lệ mong muốn sử dụng máy
            random_seed=True # Dữ liệu ngẫu nhiên
        )

        ''' BƯỚC 4: Khởi tạo máy và workcenter '''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)  # Mỗi workcenter nhận thông tin về các job

        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)          # Xác định workcenter của máy
            m.initialization(
                self.m_list,
                self.wc_list,
                self.job_creator,
                self.wc_list[wc_idx]            # Gắn máy vào đúng workcenter
            )

        ''' BƯỚC 5: Gắn DRL "brain" để học thứ tự xử lý job tại từng máy '''
        self.sqc_brain = brain_machine_S.sequencing_brain(
            self.env,
            self.job_creator,
            self.m_list,
            self.m_list,           # Máy đóng vai trò agent và cả đối tượng để huấn luyện
            self.span / 10,        # Khoảng thời gian giữa các lần cập nhật DRL (training interval)
            self.span,             # Tổng thời gian mô phỏng
            MC = 1,                # Số lần chạy Monte Carlo (có thể là lần lặp hoặc episode)
            reward_function = 1    # Hàm phần thưởng được chọn (1 có thể là "minimize tardiness", v.v.)
        )

        ''' BƯỚC 6: Chạy mô phỏng '''
        env.run()
        self.sqc_brain.check_parameter()  # Kiểm tra các tham số huấn luyện cuối cùng

# === Chạy mô hình shopfloor với tham số đã định nghĩa ===
env = simpy.Environment()          # Môi trường mô phỏng
span = 100000                      # Tổng thời gian mô phỏng
m_no = 9                           # Số máy
wc_no = 3                          # Số workcenter
spf = shopfloor(env, span, m_no, wc_no)  # Khởi chạy mô hình shopfloor
