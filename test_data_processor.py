def read_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line.split(","))
    return data[0], data[1:]

def write_file(filename, title, data):
    import csv
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(title)
        writer.writerows(data)

def sort_by_time(data):
    sorted_data = sorted(data, key=lambda x: x[0])
    ret = []
    for data in sorted_data:
        if data[-1] == "unavailable" or data[-1] == "unknown":
            continue
        ret.append(data)

    return ret

def state_statistic(data, device_state = {}):
    for d in data:
        device, state = d[1], d[2]
        if not device_state.get(device):
            device_state[device] = []
        device_state[device].append(state)
    return device_state

def is_float(data):
    try:
        float(data)
        return True
    except:
        return False

def state_value_mapping(device_state):
    import numpy as np
    device_state_value = {}
    for d, states in device_state.items():
        device_state_value[d] = {}
        if len(states) == 0:
            continue
        if is_float(states[0]):
            numeric_states = np.array(states, dtype='<U11').astype(np.float16)
            max = numeric_states.max()
            min = numeric_states.min()
            diff = max-min
            device_state_value[d]["min"] = min
            device_state_value[d]["diff"] = diff
        else:
            states_set = sorted(list(set(states)))
            states_dict = {states_set[i]: i for i in range(len(states_set))}
            states_num = len(states_set)
            for i in range(len(states)):
                device_state_value[d][states[i]] = float(states_dict[states[i]])/states_num

    return device_state_value

 
home1_device_state = {
"automation.cameraOffWhenAtHome": [],
"automation.cameraOnWhenUserLeave": [],
"automation.lightsOffWhenTooBright": [],
"automation.lightsOnWhenMotionDetected": [],
"camera.status": [],
"ceilingLamp": [],
"co2.status": [],
"co2.value": [],
"deskLamp": [],
"door": [],
"humidity": [],
"ipCamera.LightLevel": [],
"ipCamera.motion": [],
"ipCamera.motionActive": [],
"ipCamera.sound": [],
"person.home": [],
"phone.activity": [],
"phone.atHome": [],
"phone.charging": [],
"phone.sleepConfidence": [],
"phone.wifiConnection": [],
"rolo.position": [],
"sun": [],
"temperature": [],
"thermostat.heatingTemperature": [],
"thermostat.measuredTemperature": [],
"weather.homeLocation": [],
"weather.town": [],
"window": []
 }

    
def preprocess_data(test_filename, save_path='Home1/train_data.csv'):
    import os
    import numpy as np
    raw_train_data = []
    processed_test_data = []
    dir_list = os.listdir("Home1")
    for i in range(7):
        filename = dir_list[i]
        print(filename)
        _, cur_data = read_file(f"Home1/{filename}")
        raw_train_data.extend(cur_data)
    timed_raw_train_data = sort_by_time(raw_train_data)
    device_state = state_statistic(timed_raw_train_data, home1_device_state)
    device_state_value = state_value_mapping(device_state)
    device_num = len(device_state.keys())
    device_list = list(device_state.keys())
    device_vector_map = {device_list[i]:i for i in range(len(device_list))}

    _, test_data = read_file(test_filename)
    timed_raw_test_data = sort_by_time(test_data)
    data_len = len(timed_raw_test_data)

    prev_data_vector = np.zeros(device_num)
    data_vector = np.ones(device_num)
    not_change_count = 0
    for i in range(len(timed_raw_test_data)):
        cur_data = timed_raw_test_data[i]
        cur_d = cur_data[1]
        cur_value = cur_data[2]
        if is_float(cur_value):
            s = float(cur_value)
            diff = device_state_value[cur_d]["diff"]
            min = device_state_value[cur_d]["min"]
            norm_value = int(((s-min)/diff*100)/10)/10.0
        else:
            norm_value = device_state_value[cur_d][cur_value]
        vector_id = device_vector_map[cur_d]
        data_vector[vector_id] = norm_value
        temp_idx = i
        while temp_idx+1<data_len and cur_data[0] == timed_raw_test_data[temp_idx+1][0]:
            temp_idx += 1
            cur_data = timed_raw_test_data[temp_idx]
            cur_d = cur_data[1]
            cur_value = cur_data[2]
            if is_float(cur_value):
                s = float(cur_value)
                diff = device_state_value[cur_d]["diff"]
                min = device_state_value[cur_d]["min"]
                norm_value = int(((s-min)/diff*100)/10)/10.0
            else:
                norm_value = device_state_value[cur_d][cur_value]
            vector_id = device_vector_map[cur_d]
            data_vector[vector_id] = norm_value
        i = temp_idx
        processed_test_data.append(data_vector.copy())
        diff = data_vector - prev_data_vector
        if max(abs(diff)) <= 0:
            not_change_count += 1
        prev_data_vector = data_vector.copy()
    test_samples = np.array(processed_test_data)
    print(test_samples.shape, not_change_count)
    np.savetxt(save_path, test_samples, delimiter=',')
    

import argparse
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('-f', '--file', required=False, type=str)
parser.add_argument('-s', '--save', required= True, type=str)

args = parser.parse_args()


# preprocess_home1_training_data()
# preprocess_home1_training_data(save_path="Home1/train_data_7day.csv")
preprocess_data(args.file, args.save)