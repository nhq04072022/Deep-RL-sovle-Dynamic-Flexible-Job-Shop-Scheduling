import os
import simpy
import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

# Các module dùng cho mô hình hóa máy, trung tâm làm việc, luật xếp việc và định tuyến
import agent_machine
import agent_workcenter
import sequencing
import routing
import job_creation
import breakdown_creation
import heterogeneity_creation
import validation_S
import validation_R

'''
Mô phỏng hệ thống sản xuất với các tác tử định tuyến độc lập (Independent Routing Agents)
'''

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        # Bước 1: Khởi tạo môi trường mô phỏng và thông số cơ bản
        self.env = env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        m_per_wc = int(self.m_no / self.wc_no)

        # Bước 2.1: Khởi tạo các máy và lưu vào danh sách máy
        for i in range(m_no):
            machine_instance = agent_machine.machine(env, i, print=0)
            self.m_list.append(machine_instance)

        # Bước 2.2: Khởi tạo các trung tâm làm việc, mỗi trung tâm gồm nhiều máy
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            wc_instance = agent_workcenter.workcenter(env, i, x)
            self.wc_list.append(wc_instance)
            cum_m_idx += m_per_wc

        # Bước 3: Khởi tạo bộ tạo công việc
        if 'seed' in kwargs:
            self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, [10, 20], 2, 0.9, seed=kwargs['seed'])
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        # Bước 4: Gán job_creator cho các workcenter và máy
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])

        # Bước 5: Thiết lập luật xếp việc hoặc định tuyến nếu có truyền vào
        if 'sequencing_rule' in kwargs:
            rule_name = kwargs['sequencing_rule']
            try:
                sequencing_rule_fn = getattr(sequencing, rule_name)
            except AttributeError:
                raise Exception(f"Invalid sequencing rule: {rule_name}")
            for m in self.m_list:
                m.job_sequencing = sequencing_rule_fn

        if 'routing_rule' in kwargs:
            rule_name = kwargs['routing_rule']
            try:
                routing_rule_fn = getattr(routing, rule_name)
            except AttributeError:
                raise Exception(f"Invalid routing rule: {rule_name}")
            for wc in self.wc_list:
                wc.job_routing = routing_rule_fn

        # Bước 6: Nếu dùng DRL thì khởi tạo tác tử định tuyến DRL
        if 'arch' in kwargs:
            arch_flag = kwargs['arch']
            self.routing_brain = validation_R.DRL_routing(
                self.env,
                self.job_creator,
                self.wc_list,
                **{'validated': True} if arch_flag == 'validated' else {'TEST': True} if arch_flag == 'TEST' else {'global_reward': kwargs.get('global_reward', False)}
            )

    def simulation(self):
        # Chạy mô phỏng
        self.env.run()

def run_experiments():
    # Khởi tạo các biến lưu kết quả
    spf_dict = {}
    production_record = {}
    benchmark = ['EA', 'CT', 'ET', 'TT', 'UT', 'SQ']
    DRLs = ['validated']
    reward_mechanism = [False]
    title = benchmark + ['DRL_RA']
    span = 100000
    m_no = 9
    wc_no = 3
    sum_record = []
    benchmark_record = []
    max_record = []
    rate_record = []
    iteration = 1
    export_result = True

    last_spf = None
    for run in range(iteration):
        print(f'******************* ITERATION-{run} *******************')
        sum_record.append([])
        benchmark_record.append([])
        max_record.append([])
        rate_record.append([])
        seed = np.random.randint(2000000000)

        # Chạy mô phỏng với các luật định tuyến truyền thống
        for idx, rule in enumerate(benchmark):
            env = simpy.Environment()
            spf = shopfloor(env, span, m_no, wc_no, routing_rule=rule, seed=seed)
            spf.simulation()
            _, cumulative_tard, _, tard_max, tard_rate = spf.job_creator.tardiness_output()
            sum_record[run].append(cumulative_tard[-1])
            benchmark_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)
            last_spf = spf

        # Chạy mô phỏng với DRL
        for idx, x in enumerate(DRLs):
            env = simpy.Environment()
            spf = shopfloor(env, span, m_no, wc_no, arch=x, global_reward=reward_mechanism[idx], seed=seed)
            spf.simulation()
            _, cumulative_tard, _, tard_max, tard_rate = spf.job_creator.tardiness_output()
            sum_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)
            last_spf = spf

    # Hiển thị bảng kết quả tổng hợp
    print('-------------- Complete Record --------------')
    print(tabulate(sum_record, headers=title))
    print('-------------- Average Performance --------------')

    # Tính toán hiệu suất trung bình cho benchmark (không gồm DRL)
    avg_b = np.mean(benchmark_record, axis=0)
    ratio_b = np.around(avg_b / avg_b.max() * 100, 2)
    winning_rate_b = np.zeros(len(title))
    for idx in np.argmin(benchmark_record, axis=1):
        winning_rate_b[idx] += 1
    winning_rate_b = np.around(winning_rate_b / len(sum_record) * 100, 2)

    # Tính toán hiệu suất tổng thể (có DRL)
    avg = np.mean(sum_record, axis=0)
    max_values = np.mean(max_record, axis=0)
    tardy_rate = np.around(np.mean(rate_record, axis=0) * 100, 2)
    ratio = np.around(avg / avg.min() * 100, 2)
    ranking = np.argsort(ratio)
    winning_rate = np.zeros(len(title))
    for idx in np.argmin(sum_record, axis=1):
        winning_rate[idx] += 1
    winning_rate = np.around(winning_rate / len(sum_record) * 100, 2)

    for rank_idx, rule in enumerate(ranking):
        print(f"{title[rule]}, avg.: {avg[rule]} | max: {max_values[rule]} | %: {ratio[rule]}% | tardy %: {tardy_rate[rule]}% | winning rate: {winning_rate_b[rule]}/{winning_rate[rule]}%")

    # Xuất kết quả ra file Excel nếu có chọn
    if export_result:
        df_win_rate = DataFrame([winning_rate], columns=title)
        df_sum = DataFrame(sum_record, columns=title)
        df_tardy_rate = DataFrame(rate_record, columns=title)
        df_max = DataFrame(max_record, columns=title)
        df_before_win_rate = DataFrame([winning_rate_b], columns=title)
        address = os.path.join(os.getcwd(), 'experiment_result', 'RAW_RA_val.xlsx')
        Excelwriter = pd.ExcelWriter(address, engine="xlsxwriter")
        dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
        sheetname = ['win rate', 'sum', 'tardy rate', 'maximum', 'before win rate']

        for i, df in enumerate(dflist):
            df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
        Excelwriter.close()
        print(f'export to {address}')

    # Kiểm tra cấu hình của DRL routing agent nếu có
    if last_spf is not None and hasattr(last_spf, 'routing_brain'):
        last_spf.routing_brain.check_parameter()


if __name__ == "__main__":
    run_experiments()