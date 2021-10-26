import numpy as np
import math

#waiting_time_per_lane =  [1.0, 0.0, 0.0, 0.0, 1304.0, 1194.0, 1056.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1298.0, 1118.0, 1036.0, 0.0]
#out_lanes = ['TL2N_3', 'TL2E_3', 'TL2S_0', 'TL2W_2', 'TL2W_0', 'TL2N_1', 'TL2S_1', 'TL2N_0', 'TL2E_1', 'TL2W_3', 'TL2E_2', 'TL2E_0', 'TL2S_2', 'TL2W_1', 'TL2N_2', 'TL2S_3']
#lanes = ['N2TL_0', 'N2TL_1', 'N2TL_2', 'N2TL_3', 'E2TL_0', 'E2TL_1', 'E2TL_2', 'E2TL_3', 'S2TL_0', 'S2TL_1', 'S2TL_2', 'S2TL_3', 'W2TL_0', 'W2TL_1', 'W2TL_2', 'W2TL_3']
#getlaststepvehiclenumber = lanes[8, 1, 1, 2, 14, 14, 14, 2, 6, 6, 2, 1, 20, 17, 15, 1]
#getlaststepvehiclenumber = out_lanes[0, 0, 9, 1, 3, 3, 6, 8, 2, 0, 1, 4, 3, 4, 2, 2]



#waiting_time_per_lane = [35.0, 0.0, 0.0, 156.0, 7.0, 2.0, 0.0, 1.0, 26.0, 22.0, 8.0, 166.0, 0.0, 0.0, 0.0, 41.0]


getlaststepvehiclenumber = [8, 1, 1, 2, 14, 14, 14, 2, 6, 6, 2, 1, 20, 17, 15, 1]
waiting_time_per_lane =  [1.0, 0.0, 0.0, 0.0, 1304.0, 1194.0, 1056.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1298.0, 1118.0, 1036.0, 0.0]


waiting_time_per_lane = [0.0, 0.0, 0.0, 7.0, 117.0, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 195.0, 55.0, 33.0, 0.0]
vehicle_number_per_lane = [0, 0, 0, 1, 4, 1, 1, 1, 0, 2, 0, 1, 6, 3, 2, 0]
average_sped = [0, 0, 0, 0.0, 23.60310026706957, 10.0, 0, 27.248231909322854, 14.79342361570752, 0, 27.456183982196258, 0, 15.816712767147465, 26.666716817818703, 0.0, 0]


print("tru")


timings = np.random.weibull(2, 10000)
timings = np.sort(timings)

# reshape the distribution to fit the interval 0:max_steps
car_gen_steps = []
min_old = math.floor(timings[1])
max_old = math.ceil(timings[-1])
min_new = 0
max_new = 40000
for value in timings:
    car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

car_gen_steps = np.rint(car_gen_steps) 

import matplotlib.pyplot as plt

plt.plot(car_gen_steps, list(range(len(car_gen_steps))))
plt.show()


print("tru")