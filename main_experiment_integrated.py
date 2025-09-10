import os
import simpy
import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

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
        """Create environment instances and specify simulation span."""
        self.env = env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []

        m_per_wc = int(self.m_no / self.wc_no)

        # STEP 2.1: create instances of machines
        for i in range(m_no):
            machine_instance = agent_machine.machine(env, i, print=0)
            self.m_list.append(machine_instance)

        # STEP 2.2: create instances of work centers
        cum_m_idx = 0
        for i in range(wc_no):
            machines_for_wc = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            wc_instance = agent_workcenter.workcenter(env, i, machines_for_wc)
            self.wc_list.append(wc_instance)
            cum_m_idx += m_per_wc

        # STEP 3: initialize the job creator
        if 'seed' in kwargs:
            self.job_creator = job_creation.creation(
                self.env,
                self.span,
                self.m_list,
                self.wc_list,
                kwargs['pt_range'],
                kwargs['tightness'],
                0.9,
                seed=kwargs['seed'],
            )
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        # STEP 4: initialize machines and work centers
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])

        # STEP 5: set sequencing or routing rules, and DRL
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

        # specify the framework of DRL
        if kwargs.get('DRL_AS'):
            print("---> Integrated DRL mode ON <---")
            kwargs['DRL_R'] = True
            kwargs['AS'] = True

        # specify the sequencing agents
        if kwargs.get('AS'):
            print("---> AS mode ON <---")
            self.sequencing_brain = validation_S.DRL_sequencing(
                self.env, self.m_list, self.job_creator, self.span, show=0, validated=1, reward_function=''
            )

        # specify the routing agents
        if kwargs.get('DRL_R'):
            print("---> DRL Routing mode ON <---")
            self.routing_brain = validation_R.DRL_routing(
                self.env, self.job_creator, self.wc_list, validated=1
            )

    def simulation(self):
        self.env.run()

# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments


benchmark_HH = [['FIFO','EA'],['AVPRO','CT'],['SPT','CT'],['LWKRSPT','CT'],['PTWINQ','CT'],['DPTWINQNPT','CT'],['AVPRO','ET'],['SPT','ET'],['LWKRSPT','ET'],['PTWINQ','ET'],['DPTWINQNPT','ET'],['GP_S1','GP_R1'],['GP_S2','GP_R2']]
benchmark_HL = [['FIFO','EA'],['ATC','CT'],['CR','CT'],['EDD','CT'],['MOD','CT'],['DPTLWKRS','CT'],['ATC','ET'],['CR','ET'],['EDD','ET'],['MOD','ET'],['DPTLWKRS','ET'],['GP_S1','GP_R1'],['GP_S2','GP_R2']]
benchmark_LH = [['FIFO','EA'],['EDD','CT'],['MOD','CT'],['MS','CT'],['PTWINQS','CT'],['DPTLWKRS','CT'],['EDD','EA'],['MOD','EA'],['MS','EA'],['PTWINQS','EA'],['DPTLWKRS','EA'],['GP_S1','GP_R1'],['GP_S2','GP_R2']]
benchmark_LL = [['FIFO','EA'],['CR','CT'],['MOD','CT'],['MS','CT'],['PTWINQS','CT'],['DPTLWKRS','CT'],['CR','EA'],['MOD','EA'],['MS','EA'],['PTWINQS','EA'],['DPTLWKRS','EA'],['GP_S1','GP_R1'],['GP_S2','GP_R2']]

def select_benchmark(pt_range, tightness):
    if pt_range[1] / pt_range[0] > 2.5:
        return benchmark_HH if tightness == 2 else benchmark_HL
    return benchmark_LH if tightness == 3 else benchmark_LL


def build_titles(benchmark):
    base_titles = [x[0] + '+' + x[1] for x in benchmark[:-2]] + ['GP1', 'GP2']
    return base_titles + ['RA alone', 'SA alone', 'Integrated DRL']


def run_experiments():
    pt_range = [10, 20]
    tightness = 3

    benchmark = select_benchmark(pt_range, tightness)

    span = 100000
    m_no = 9
    wc_no = 3

    DRLs = ['DRL_R', 'AS', 'DRL_AS']
    title = build_titles(benchmark)
    print(title)

    sum_record = []
    benchmark_record = []
    max_record = []
    rate_record = []
    iteration = 5
    export_result = True
    print(tightness)

    last_spf = None
    for run in range(iteration):
        print('******************* ITERATION-{} *******************'.format(run))
        sum_record.append([])
        benchmark_record.append([])
        max_record.append([])
        rate_record.append([])
        seed = np.random.randint(2000000000)

        # run simulation with different rules
        for idx, rule in enumerate(benchmark):
            env = simpy.Environment()
            spf = shopfloor(
                env,
                span,
                m_no,
                wc_no,
                pt_range=pt_range,
                tightness=tightness,
                sequencing_rule=rule[0],
                routing_rule=rule[1],
                seed=seed,
            )

            # add breakdowns
            breakdown_creation.creation(env, spf.m_list, [0, 3, 6], [10000, 20000, 30000], [100, 150, 120])

            # add heterogeneity changes
            heterogeneity_creation.creation(env, spf.job_creator, [170, 970], [[10, 15], [10, 30]])

            spf.simulation()
            _, cumulative_tard, _, tard_max, tard_rate = spf.job_creator.tardiness_output()
            sum_record[run].append(cumulative_tard[-1])
            benchmark_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)
            last_spf = spf

        # extra runs with DRL
        for idx, x in enumerate(DRLs):
            env = simpy.Environment()
            flags = {x: True}
            spf = shopfloor(
                env,
                span,
                m_no,
                wc_no,
                pt_range=pt_range,
                tightness=tightness,
                seed=seed,
                **flags,
            )
            spf.simulation()
            _, cumulative_tard, _, tard_max, tard_rate = spf.job_creator.tardiness_output()
            sum_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)
            last_spf = spf

    print('-------------- Complete Record --------------')
    print(tabulate(sum_record, headers=title))
    print('-------------- Average Performance --------------')

    # performance without DRL
    avg_b = np.mean(benchmark_record, axis=0)
    ratio_b = np.around(avg_b / avg_b.max() * 100, 2)
    winning_rate_b = np.zeros(len(title))
    for idx in np.argmin(benchmark_record, axis=1):
        winning_rate_b[idx] += 1
    winning_rate_b = np.around(winning_rate_b / len(sum_record) * 100, 2)

    # overall performance (include DRL)
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
        print(
            "{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%".format(
                title[rule], avg[rule], max_values[rule], ratio[rule], tardy_rate[rule], winning_rate_b[rule], winning_rate[rule]
            )
        )

    if export_result:
        df_win_rate = DataFrame([winning_rate], columns=title)
        df_sum = DataFrame(sum_record, columns=title)
        df_tardy_rate = DataFrame(rate_record, columns=title)
        df_max = DataFrame(max_record, columns=title)
        df_before_win_rate = DataFrame([winning_rate_b], columns=title)
        address = os.path.join(os.getcwd(), 'experiment_result', 'Integrated_experiment.xlsx')
        Excelwriter = pd.ExcelWriter(address, engine="xlsxwriter")
        dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
        sheetname = ['win rate', 'sum', 'tardy rate', 'maximum', 'before win rate']

        for i, df in enumerate(dflist):
            df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
        Excelwriter.close()
        print('export to {}'.format(address))

    # check the parameter and scenario setting for the last run if available
    if last_spf is not None:
        if hasattr(last_spf, 'routing_brain'):
            last_spf.routing_brain.check_parameter()
        if hasattr(last_spf, 'sequencing_brain'):
            last_spf.sequencing_brain.check_parameter()


if __name__ == "__main__":
    run_experiments()