import os
import simpy
import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

import job_creation
import validation_S
import agent_machine
import agent_workcenter
import sequencing
import routing
import job_creation
import breakdown_creation
import heterogeneity_creation
import validation_S
import validation_R

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        m_per_wc = int(self.m_no / self.wc_no)

        for i in range(m_no):
            machine_instance = agent_machine.machine(env, i, print = 0)
            self.m_list.append(machine_instance)

        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            wc_instance = agent_workcenter.workcenter(env, i, x)
            self.wc_list.append(wc_instance)
            cum_m_idx += m_per_wc

        if 'seed' in kwargs:
            self.job_creator = job_creation.creation(
                self.env, self.span, self.m_list, self.wc_list, [5,25], 2, 0.9, seed=kwargs['seed']
            )
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

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

        if kwargs.get('DRL_S'):
            print("---> DRL Sequencing mode ON <---")
            self.sequencing_brain = validation_S.DRL_sequencing(
                self.env, self.m_list, self.job_creator, show=0,  validated=1, reward_function=''
            )

    def simulation(self):
        self.env.run()

# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments
benchmark = ['FIFO','ATC','AVPRO','COVERT','CR','EDD','LWKR','MDD','MOD','MON','MS','NPT','SPT','WINQ','LWKRSPT','LWKRMOD','PTWINQ','PTWINQS','DPTLWKRS','DPTWINQNPT']
title = benchmark + ['DRL_SA']
span = 100000
m_no = 9
wc_no = 3
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 10
export_result = True

for run in range(iteration):
    print('******************* ITERATION-{} *******************'.format(run))
    sum_record.append([])
    benchmark_record.append([])
    max_record.append([])
    rate_record.append([])
    seed = np.random.randint(2000000000)
    # run simulation with different rules
    for idx,rule in enumerate(benchmark):
        env = simpy.Environment()
        spf = shopfloor(env, span, m_no, wc_no, sequencing_rule = rule, seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        benchmark_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)
    env = simpy.Environment()
    spf = shopfloor(env, span, m_no, wc_no, DRL_S=True, seed = seed)
    spf.simulation()
    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
    sum_record[run].append(cumulative_tard[-1])
    max_record[run].append(tard_max)
    rate_record[run].append(tard_rate)
    print('Number of jobs created',spf.job_creator.total_no)

print('-------------- Complete Record --------------')
print(tabulate(sum_record, headers=title))
print('-------------- Average Performance --------------')

# get the performnce without DRL
avg_b = np.mean(benchmark_record,axis=0)
ratio_b = np.around(avg_b/avg_b.max()*100,2)
winning_rate_b = np.zeros(len(title))
for idx in np.argmin(benchmark_record,axis=1):
    winning_rate_b[idx] += 1
winning_rate_b = np.around(winning_rate_b/iteration*100,2)

# get the overall performance (include DRL)
avg = np.mean(sum_record,axis=0)
max = np.mean(max_record,axis=0)
tardy_rate = np.around(np.mean(rate_record,axis=0)*100,2)
ratio = np.around(avg/avg.min()*100,2)
rank = np.argsort(ratio)
winning_rate = np.zeros(len(title))
for idx in np.argmin(sum_record,axis=1):
    winning_rate[idx] += 1
winning_rate = np.around(winning_rate/iteration*100,2)
for rank,rule in enumerate(rank):
    print(
        "{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%"
        .format(title[rule], avg[rule], max[rule], ratio[rule], tardy_rate[rule], winning_rate_b[rule], winning_rate[rule])
        )

if export_result:
    df_win_rate = DataFrame([winning_rate], columns=title)
    #print(df_win_rate)
    df_sum = DataFrame(sum_record, columns=title)
    #print(df_sum)
    df_tardy_rate = DataFrame(rate_record, columns=title)
    #print(df_tardy_rate)
    df_max = DataFrame(max_record, columns=title)
    #print(df_max)
    df_before_win_rate = DataFrame([winning_rate_b], columns=title)
    address = os.path.join(os.getcwd(), 'experiment_result', 'RAW_SA_experiment1.xlsx')
    Excelwriter = pd.ExcelWriter(address, engine = "xlsxwriter")
    dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
    sheetname = ['win rate', 'sum', 'tardy rate', 'maximum', 'before win rate']

    for i,df in enumerate(dflist):
        df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
    Excelwriter.close()
    print('export to {}'.format(address))

# check the parameter and scenario setting (if DRL was used)
if 'spf' in locals() and hasattr(spf, 'sequencing_brain'):
    spf.sequencing_brain.check_parameter()