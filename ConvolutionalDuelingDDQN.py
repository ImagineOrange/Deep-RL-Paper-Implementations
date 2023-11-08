class Dueling_DoubleDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        #nn.Conv2d modules expect a 4-dimensional tensor with the shape [batch_size, channels, height, width] 
        self.conv_features = nn.Sequential(
        nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=1),
        nn.ReLU()
    )  

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape),256),
            nn.ReLU(),
            nn.Linear(256,n_actions)
    )

        self.value_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape),256),
            nn.ReLU(),
            nn.Linear(256,1)
    )

    
    def forward(self, x):
        batch_size = x.size(0)
        #get features from convolutional layer
        conv_out = self.conv_features(x).view(x.size()[0], -1)
        #value from value stream
        value = self.value_stream(conv_out)
        #advantages from advantage stream
        actions_advantages = self.advantage_stream(conv_out)
        #average advantage
        average_advantages  = torch.mean(actions_advantages, dim=1, keepdim=True)

        Q = value + (actions_advantages - average_advantages)

        #return Q values
        return Q
    
    def get_conv_out(self, input_shape):
         o = self.conv_features(torch.zeros(1, * input_shape))
         o = int(np.prod(o.size()))
         return o
