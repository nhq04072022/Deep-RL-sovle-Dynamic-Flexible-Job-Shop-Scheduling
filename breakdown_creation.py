import numpy as np

'''
Mô-đun mô phỏng các sự cố hỏng hóc của máy trong hệ thống sản xuất (có thể bật/tắt tùy chọn).
'''

class creation:
    def __init__(self, env, machine_list, target_index, event_intervals, duration, **kwargs):
        self.env = env
        self.m_list = machine_list                    # Danh sách các máy trong hệ thống
        self.target_index = target_index              # Danh sách chỉ số máy sẽ bị hỏng
        self.event_intervals = event_intervals        # Khoảng thời gian giữa các sự cố
        self.duration = duration                      # Thời gian máy bị hỏng

        # Tính thời điểm bắt đầu và kết thúc mỗi sự cố
        self.event_start_time = np.cumsum(self.event_intervals)
        self.event_end_time = self.event_start_time + duration

        # Chuyển sang list để dễ sử dụng pop()
        self.event_start_time = self.event_start_time.tolist()
        self.event_end_time = self.event_end_time.tolist()

        print("The event start time: %s\nThe event end time: %s" % (self.event_start_time, self.event_end_time))
        print("--------------------------------------")
        
        self.event_number = len(target_index)
        
        # Kiểm tra số lượng cấu hình hợp lệ
        if len(target_index) != len(event_intervals):
            print('Unmatching size of events')
            raise KeyError

        # Bắt đầu tiến trình xử lý sự cố
        self.env.process(self.manipulation())

    def manipulation(self):
        for i in range(self.event_number):
            # Chờ đến thời điểm xảy ra sự cố
            yield self.env.timeout(self.event_intervals.pop(0))

            # Lấy thông tin máy bị hỏng, thời gian hỏng, và thời điểm khôi phục
            idx = self.target_index.pop(0)
            duration = self.duration.pop(0)
            restart_time = self.event_end_time.pop(0)

            # Kích hoạt tiến trình mô phỏng sự cố máy
            self.env.process(self.event_process(idx, duration, restart_time))

    def event_process(self, idx, duration, restart_time):
        # Đặt trạng thái máy sang 'không hoạt động' bằng event
        self.m_list[idx].working_event = self.env.event()

        # Ghi lại thời điểm sẽ khôi phục
        self.m_list[idx].restart_time = restart_time

        # Máy bị dừng hoạt động trong một khoảng thời gian
        yield self.env.timeout(duration)

        # Máy hoạt động trở lại sau khi hết thời gian
        self.m_list[idx].restart_time = 0
        self.m_list[idx].working_event.succeed()
