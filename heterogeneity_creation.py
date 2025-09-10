import numpy as np

'''
Thay đổi thiết lập mô phỏng (ví dụ: độ dị biệt của thời gian xử lý) trong quá trình chạy.
'''

class creation:
    def __init__(self, env, target, event_intervals, pt_range_list, **kwargs):
        self.env = env                                   # Môi trường mô phỏng (SimPy)
        self.target = target                             # Đối tượng bị thay đổi cấu hình, thường là job_creator
        self.event_intervals = list(event_intervals)     # Danh sách khoảng thời gian giữa các lần thay đổi
        self.pt_range_list = list(pt_range_list)         # Danh sách các pt_range mới (VD: [5, 15], [10, 30], ...)
        
        print('Durations and pt:', event_intervals, pt_range_list)
        print("--------------------------------------")
        
        # Kiểm tra: số lượng lần thay đổi và số pt_range phải khớp nhau
        if len(self.event_intervals) != len(self.pt_range_list):
            print('Unmatching size of events')
            raise KeyError
        
        # Bắt đầu một tiến trình song song để thực hiện việc thay đổi
        self.env.process(self.manipulation())

    def manipulation(self):
        while len(self.event_intervals):
            # In ra trạng thái hiện tại
            print("Time {}, change the heterogenity of arriving jobs to: {}".format(self.env.now, self.pt_range_list[0]))
            
            # Gọi hàm thay đổi cấu hình trong đối tượng target (job_creator)
            self.target.change_setting(self.pt_range_list.pop(0))
            
            # Đợi đến mốc thời gian tiếp theo
            yield self.env.timeout(self.event_intervals.pop(0))
