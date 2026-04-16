# %%
import numpy as np

# %%
class environment:
    def __init__(self, grid_height, grid_width):
        self.height = grid_height
        self.width = grid_width
        self.start = []
        self.end = []
        self.reward = []
        self.map = np.array([i for i in range(grid_height * grid_width)])
        self.action_space = [0,1,2,3]
    def get_Map(self):
        print(self.map.reshape([self.width, self.height]))
    def get_NumState(self):
        return self.height * self.width
    def map_Designate(self, start_cell, end_cell, reward):
        self.start.append(start_cell)
        self.end.append(end_cell)
        self.reward.append(reward)
    def get_Observation(self, location, action):
        # giả định tại vị trí nào đó với hành động gì đó thì điều gì xảy ra
        if action == -1: # không làm gì cả đứng yên quan sát môi trường
            return None, self.action_space, None
        new_location = 0
        if location in self.start: # nếu như vị trí đã biết
            idx = self.start.index(location)
            new_location = self.end[idx]
            reward = self.reward[idx]
            return new_location, self.action_space, reward

        reward = 0
        # đối các vị trí bình thường không có portal
        if action == 0:
            if location - self.width < 0:
            # nếu như nằm ở hàng đầu tiên không tiến lên được nữa
                new_location = location
            else:
                new_location = location - self.width
        elif action == 1: 
            if location + self.width > self.height * self.width - 1:
                new_location = location
            else:
                new_location = location + self.width
        elif action == 2: 
            if location % self.width == 0:
                new_location = location 
            else:
                new_location = location - 1
        elif action == 3: 
            if (location + 1) % self.width == 0:
                new_location = location 
            else:
                new_location = location + 1
        return new_location, self.action_space, reward

# %%
Envir = environment(8,8)
Envir.get_Map()
Envir.map_Designate(17,56,-15)
Envir.map_Designate(18,56,-15)
Envir.map_Designate(19,56,-15)
Envir.map_Designate(21,56,-15)
Envir.map_Designate(25,56,-15)
Envir.map_Designate(33,56,-15)
Envir.map_Designate(41,56,-15)
Envir.map_Designate(42,56,-15)
Envir.map_Designate(43,56,-15)
Envir.map_Designate(46,56,-15)
Envir.map_Designate(47,56,-15)
Envir.map_Designate(47,56,-15)
Envir.map_Designate(15,56,+15)
Envir.map_Designate(1,10,+5)
Envir.map_Designate(26,56,+20)

if __name__ == 'main':
    for i in range(len(Envir.start)):
        print('i = '+ str(i) + '|Start at ' + str(Envir.start[i]) + ' results at ' + str(Envir.end[i]) + ' get Reward: ' + str(Envir.reward[i]))

# %% [markdown]
# getAction ở dưới có tác dụng là đi đến vị trí với hành động được chọn ở bước trước và đồng thời trả về hành động nên làm ở bước hiện tại

# %%
class MAB_agent:
    def __init__(self, envir, init_location, lr, gamma):
        self.envir = envir
        self.rewards = []
        self.cur_location = init_location
        # cần lưu lại value của các trạng thái với các hành động tương ứng
        self.q_table = {}
        self.last_action = None
        self.lr = lr
        self.gamma = gamma

    def getTotalReward(self):
        return np.sum(self.rewards)
    
    def getAction(self, observation):
        # đến vị trí với action được chọn ở bước trước
        location_now, action_space, pre_reward = observation
        
        if pre_reward is not None and self.last_action is not None:
            # kiểm tra xem location hiện tại có trong q table chưa
            if self.cur_location not in self.q_table:
                self.q_table[self.cur_location] = [0.0] * len(action_space)
            if location_now not in self.q_table:
                self.q_table[location_now] = [0.0] * len(action_space)
            self.q_table[self.cur_location][self.last_action] = (1-self.lr) * self.q_table[self.cur_location][self.last_action] + self.lr * (pre_reward + self.gamma * np.max(self.q_table[location_now]))
            self.rewards.append(pre_reward)
        
        if location_now is not None:
            # gán cho cur location bằng với vị trị hiện tại trước rồi sau đó quyết định nên đi tiếp thế nào bằng cách chọn ra 1 action
            self.cur_location = location_now
                
        if self.cur_location in self.q_table:
            max_val = np.max(self.q_table[self.cur_location])
            idx_candidate = []
            for idx in range(len(self.q_table[self.cur_location])):
                if self.q_table[self.cur_location][idx] == max_val:
                    idx_candidate.append(idx)
            if len(idx_candidate) >= 2:
                action = np.random.choice(idx_candidate)
            else:
                action = np.argmax(self.q_table[self.cur_location])
        else:
            action = action_space[np.random.randint(low = 0, high = len(action_space))]
        
        self.last_action = action
        return action

# %%
class MABe_agent(MAB_agent):
    def __init__(self, envir, init_location, epsilon, lr, gamma):
        super().__init__(envir, init_location, lr, gamma)
        self.threshold_greedy = epsilon
        self.last_action = None
    def getAction(self, observation):
        # đến vị trí với action được chọn ở bước trước
        location_now, action_space, pre_reward = observation
        
        if pre_reward is not None and self.last_action is not None:
            # kiểm tra xem location hiện tại có trong q table chưa
            if self.cur_location not in self.q_table:
                self.q_table[self.cur_location] = [0.0] * len(action_space)
            if location_now not in self.q_table:
                self.q_table[location_now] = [0.0] * len(action_space)
            self.q_table[self.cur_location][self.last_action] = (1-self.lr) * self.q_table[self.cur_location][self.last_action] + self.lr * (pre_reward + self.gamma * np.max(self.q_table[location_now]))
            self.rewards.append(pre_reward)
        
        if location_now is not None:
            # gán cho cur location bằng với vị trị hiện tại trước rồi sau đó quyết định nên đi tiếp thế nào bằng cách chọn ra 1 action
            self.cur_location = location_now
        
        rand_val = np.random.rand()
        if rand_val < self.threshold_greedy:
            action = action_space[np.random.randint(low = 0, high = len(action_space))]
        else: 
            if self.cur_location in self.q_table:
                action = np.argmax(self.q_table[self.cur_location])
            else:
                action = action_space[np.random.randint(low = 0, high = len(action_space))]
        
        self.last_action = action
        return action
# %%
