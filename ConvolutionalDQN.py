class DQN(nn.Module):
    def __init__(self, input_shape, n_actions ,epsilon_start):
        super().__init__()

        #experience replay
        self.replay_memory = deque([],maxlen=10000)
        self.epsilon_start = epsilon_start

        #nn.Conv2d modules expect a 4-dimensional tensor with the shape [batch_size, channels, height, width] - (input_shape[0])
        self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=1),
        nn.ReLU()
    )  

        self.dense = nn.Sequential(
        nn.Linear(self.get_conv_out(input_shape), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,n_actions),
    )

  
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.dense(conv_out)
    

    def get_conv_out(self, input_shape):
         o = self.conv(torch.zeros(1, * input_shape))
         o = int(np.prod(o.size()))
         return o
    
    def epsilon_greedy(self,frame,observation):
        self.frame = frame
        self.current_dir = current_dir
       
       #decay as a function of frames
        self.epsilon = self.epsilon_start * (1 / (1 + 1e-6 * self.frame))
        self.rand = random.uniform(0, 1)

        if self.rand <= self.epsilon:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action)
        return self.forward(observation)


class Experience_Replay: #Experience Replay mechanism for model training
    def __init__(self, capacity):
        #initialize deque of maxlen capacity
        self.buffer = deque(maxlen=capacity)
    
    def append(self, experience):
        #append experience stack to deque
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        #random indices of deque
        indices = np.random.default_rng().choice(len(self.buffer), batch_size, replace=False)
        #the zip + * op is going to iterate over all data types in the deque 
        states_1, states_2, actions, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        #return np arrays of unpacked data
        return np.array(states_1), np.array(states_2), np.array(actions), np.array(rewards), np.array(dones)
