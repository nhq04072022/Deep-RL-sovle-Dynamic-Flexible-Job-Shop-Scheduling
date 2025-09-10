# các thư viện cần thiết 
import simpy                      # Thư viện mô phỏng (Discrete Event Simulation)
import sys
sys.path                          # Có thể dùng để kiểm tra hoặc thêm đường dẫn tới module tùy chỉnh

import matplotlib.pyplot as plt   
import matplotlib.animation as animation  
import torch                      
import numpy as np               # Xử lý số liệu
from tabulate import tabulate    # Hiển thị bảng kết quả 

# === Nhập các module dùng cho mô hình hóa máy, trung tâm làm việc, luật xếp việc và định tuyến ===
import agent_machine              # Định nghĩa agent cho máy
import agent_workcenter          # Định nghĩa agent cho trung tâm làm việc (workcenter)
import brain_workcenter_R        # DRL brain cho routing agent (định tuyến công việc)
import job_creation              # Module tạo công việc (job)
import breakdown_creation        # Tạo sự cố breakdown ngẫu nhiên (nếu có)
import heterogeneity_creation    # Tạo khác biệt pt_range
import validation_S             

"""
MODULE DÙNG ĐỂ HUẤN LUYỆN TÁC NHÂN ĐỊNH TUYẾN (ROUTING AGENT)
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        ''' Khởi tạo sàn sản xuất gồm các máy và trung tâm làm việc '''
        self.env = env            # Môi trường mô phỏng SimPy
        self.span = span          # Tổng thời gian mô phỏng
        self.m_no = m_no          # Số lượng máy
        self.m_list = []          # Danh sách các máy
        self.wc_no = wc_no        # Số lượng trung tâm làm việc (workcenter)
        self.wc_list = []         # Danh sách các workcenter
        m_per_wc = int(self.m_no / self.wc_no)  # Số máy trên mỗi workcenter

        # === BƯỚC 1: Tạo các máy ===
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i)
            exec(expr1)  # Tạo máy agent_machine.machine và gán cho self.m_i
            expr2 = '''self.m_list.append(self.m_{})'''.format(i)
            exec(expr2)  # Thêm vào danh sách máy

        # === BƯỚC 2: Tạo các workcenter (tập hợp máy) ===
        cum_m_idx = 0
        for i in range(wc_no):
            # Lấy m_per_wc máy liên tiếp để tạo workcenter
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i)
            exec(expr1)  # Khởi tạo workcenter
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i)
            exec(expr2)  # Thêm vào danh sách workcenter
            cum_m_idx += m_per_wc

        # === BƯỚC 3: Khởi tạo bộ tạo công việc (Job Creator) ===
        # Tạo các công việc với các tham số: thời gian xử lý, độ gấp rút, mức sử dụng kỳ vọng...
        self.job_creator = job_creation.creation(
            self.env, self.span, self.m_list, self.wc_list,
            [1, 50],        # Khoảng thời gian xử lý cho mỗi job
            3,              # Mức độ gấp rút (due tightness)
            0.8,            # Kỳ vọng mức sử dụng máy
            random_seed=True  # Sinh dữ liệu ngẫu nhiên mỗi lần chạy
        )
        self.job_creator.output()  # In ra danh sách công việc đã tạo

        # === BƯỚC 4: Khởi tạo các máy và workcenter ===
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)  # Cung cấp thông tin job cho mỗi workcenter
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])

        # === BƯỚC 5: Gắn bộ não định tuyến DRL cho các workcenter ===
        self.routing_brain = brain_workcenter_R.routing_brain(
            self.env,
            self.job_creator,
            self.m_list,
            self.wc_list,
            self.span / 5,  # Khoảng thời gian delay giữa các lần huấn luyện
            self.span       # Tổng thời gian mô phỏng
        )

        # === BƯỚC 6: Chạy mô phỏng ===
        env.run()
        # Kiểm tra lại các tham số trong mô hình DRL
        self.routing_brain.check_parameter()


# === Chạy mô phỏng thực tế ===
env = simpy.Environment()  # Tạo môi trường mô phỏng
span = 100000              # Tổng thời gian mô phỏng
m_no = 9                   # Số lượng máy
wc_no = 3                  # Số lượng trung tâm làm việc
spf = shopfloor(env, span, m_no, wc_no)  # Tạo và chạy mô hình 
