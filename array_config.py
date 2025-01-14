

# sample_rate = 48000
# rows = 1
#
#
# # 12 mics
# cols = 12
# mic_spacing = 0.08  # meters - based on center freq
# num_mics = rows * cols
#
# mic_positions = [
#         (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11)
#     ]


'''
12 mics at 0.08m spacing: Center Frequency = 2144 Hz
6 mics at 0.16m spacing: Center Frequency = 1075 Hz
4 mics at 0.24m spacing: Center Frequency = 715 Hz
3 mics at 0.32m spacing: Center Frequency = 536 Hz
3 mics at 0.4m spacing: Center Frequency = 429 Hz
2 mics at 0.48m spacing: Center Frequency = 357 Hz
'''



class Array_Config:
    def __init__(self, name, cols, spacing, mic_positions):
        self.name = name
        self.sample_rate = 48000
        self.rows = 1
        self.cols = cols
        self.spacing = spacing # in meters
        self.num_mics = self.rows * self.cols
        self.mic_positions = mic_positions



array_full = Array_Config('array_full', 12, 0.08,
        [(0, 0), (0, 1), (0, 2), (0, 3),
         (0, 4), (0, 5), (0, 6), (0, 7),
         (0, 8), (0, 9), (0, 10), (0, 11)])
array_half = Array_Config('array_half', 6, 0.16,
    [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (0, 10)])
array_quarter = Array_Config('array_quarter', 4, 0.24,
    [(0, 0), (0, 3), (0, 6), (0, 9)])
array_third_1 = Array_Config('array_third_1', 3, 0.32,
    [(0, 0), (0, 4), (0, 8)])
array_third_2 = Array_Config('array_third_2', 3, 0.4,
    [(0, 0), (0, 5), (0, 10)])
array_double = Array_Config('array_double', 2, 0.48,
    [(0, 0), (0, 6)])

array_config_list = [array_full, array_half, array_quarter,
                     array_third_1, array_third_2, array_double]



if __name__ == '__main__':

    pass



