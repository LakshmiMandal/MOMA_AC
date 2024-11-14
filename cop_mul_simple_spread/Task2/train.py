import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
from transformers import BertTokenizer,BertModel
from transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 10000
testing_episodes = 1000
# episode_time = 25

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()
#         print("action is:", action)

#     env.step(action)
# env.close()

class Actor(nn.Module):
    def __init__(self, state_size, action_size, actor_lr):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Linear(self.state_size, 48)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(48, 12)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(12, self.action_size)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights using He uniform initialization
        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)
        nn.init.kaiming_uniform_(self.layer3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return Categorical(x)
        

class Critic(nn.Module):
    def __init__(self, full_state_size, value_size,critic_lr):
        super(Critic, self).__init__()
        # self.state_size = state_size
        self.value_size = value_size

        self.layer1 = nn.Linear(full_state_size, 36*7)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(36*7, 3*36)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(3*36, self.value_size)

        # Initialize weights using He uniform initialization
        # nn.init.kaiming_uniform_(self.layer1.weight)
        # nn.init.kaiming_uniform_(self.layer2.weight)
        # nn.init.kaiming_uniform_(self.layer3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        # x = self.relu4(self.layer4(x))
        x = self.layer3(x)
        return x

class B_model():
    def __init__(self):
        super(B_model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

        
        
    def bert_model(self):
        # super(B_model, self).__init__()  
        # prompt = "agent1 go to dest1 and agent2 go to dest2."
        prompt = "agent_0 go to landmark 1, agent_1 go to landmark 2, agent_2 go to landmark 0."
        encoded_input = self.tokenizer(prompt, return_tensors='pt')  
        bert_output = self.model(**encoded_input)
        # encoding = tokenizer.encode(text)
        # total_loss = model(**encoding, labels=torch.LongTensor([1])).loss
        # print("loss in BERT",total_loss)
        # Tokenize and encode the text
        return bert_output 

class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
       
        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.9
        self.actor_lr = 0.001
        # self.actor_lr = 0.0005
        # self.actor_lr = 0.0005
        self.critic_lr = 0.005

        # create model for policy network
        self.actor0 = Actor(self.state_size, self.action_size, self.actor_lr) 
        self.actor1 = Actor(self.state_size, self.action_size, self.actor_lr)
        self.actor2 = Actor(self.state_size, self.action_size, self.actor_lr) 
        path_C = "./model/critic.pkl"
        self.critic = torch.load(path_C)      
        # self.critic = Critic(3*(self.state_size)+(26*768), self.value_size, self.critic_lr)
        # self.critic1 = Critic(self.state_size, self.value_size, self.critic_lr)
        self.b_model=B_model()

    def get_action(self, state0,state1, state2):
        #Agent0
        state_tensor0 = torch.FloatTensor(state0)
        policy0 = self.actor0(state_tensor0)
        dist_samp0=policy0.sample()
        log_prob0=policy0.log_prob(dist_samp0)
        # log_prob0.requires_grad_(True)
        action0=dist_samp0.cpu().numpy()
        # print("action0",action0)

        #Agent1
        state_tensor1 = torch.FloatTensor(state1)
        policy1 = self.actor1(state_tensor1)
        dist_samp1=policy1.sample()
        log_prob1=policy1.log_prob(dist_samp1)
        # log_prob0.requires_grad_(True)
        action1=dist_samp1.cpu().numpy()
        # print("action1",action1)        

        # #Agent2
        state_tensor2 = torch.FloatTensor(state2)
        policy2 = self.actor2(state_tensor2)
        dist_samp2=policy2.sample()
        log_prob2=policy2.log_prob(dist_samp2)
        # log_prob0.requires_grad_(True)
        action2=dist_samp2.cpu().numpy()
        # # print("action2",action2)
        return log_prob0,action0, log_prob1,action1, log_prob2,action2
    
    def train_model(self, state,log_prob0,log_prob1, log_prob2,sin_stage_reward, next_state, b_hid_st, done):

        # print(state,"\n",log_prob0,"\n",log_prob1,"\n",log_prob2,"\n",sin_stage_reward,"\n", next_state, "\n",done,"\n")
        b_hid_st=b_hid_st.detach()
        state_tensor = torch.FloatTensor(state)
        # print("shape........",state_tensor.shape)
        state_tensor_b = torch.cat([state_tensor, b_hid_st], 1)
        next_state_tensor = torch.FloatTensor(next_state)
        next_state_tensor_b = torch.cat([next_state_tensor, b_hid_st], 1)
        value = self.critic(state_tensor_b)[0]
        
        # value.requires_grad_(True)
        next_value = self.critic(next_state_tensor_b)[0]

        if done:
            advantages = (sin_stage_reward - value)
            # advantages1 = (sin_stage_reward - value)
            # advantages2 = (sin_stage_reward - value)
            target= sin_stage_reward
        else:
            advantages = (sin_stage_reward + self.discount_factor * (next_value.detach()) - value)
            # advantages1 = (sin_stage_reward + self.discount_factor * (next_value.detach()) - value)
            # advantages2 = (sin_stage_reward + self.discount_factor * (next_value.detach()) - value)
            target= sin_stage_reward + self.discount_factor * (next_value.detach())
        loss_actor0= log_prob0*(advantages.detach())
        self.actor0.optimizer.zero_grad()
        loss_actor0.backward()
        self.actor0.optimizer.step()

        #detach from advantages for actor
        #Train actor1
        loss_actor1= log_prob1*(advantages.detach())
        self.actor1.optimizer.zero_grad()
        loss_actor1.backward()
        self.actor1.optimizer.step()

        # #Train actor2
        loss_actor2= log_prob2*(advantages.detach())
        self.actor2.optimizer.zero_grad()
        loss_actor2.backward()
        self.actor2.optimizer.step()

        #detach from next val for critic 

        #Train Critic
        loss_critic = F.mse_loss(value, target)
        # print("critic loss:",loss_critic)
        # loss_critic = torch.sqrt(loss)
        # loss_critic =error.pow(2)
        self.critic.optimizer.zero_grad()
        loss_critic.backward()
        self.critic.optimizer.step()

if __name__ == "__main__":
    # get size of state and action from environment
    if not os.path.exists("./model"):   
        os.makedirs("./model")
    actor0_save_path="model/actor0.pkl"
    actor1_save_path="model/actor1.pkl"
    actor2_save_path="model/actor2.pkl" 
    critic_save_path="model/critic.pkl"   
    no_agent=3
    env = simple_spread_v3.parallel_env(render_mode="human")
    state_size = no_agent*6 #24 #env.observation_space.shape[0]
    action_size = 5 #env.action_space.n
    
  
    total_runs = 1
    avg_reward=[]
    f=open("simple_spread_logFile_itr_avgCost","w+")
    # store_total_cost = np.zeros((1,EPISODES))
    for runs in range(total_runs):
            # np.random.seed((runs+1)*100)
            # random.seed((runs+1)*110)
            # make A2C agent
            agent = A2CAgent(state_size, action_size)
            bert_op=agent.b_model.bert_model()
        # bert_op_tensor=bert_op['pooler_output']
            hid_st=bert_op['last_hidden_state'].view(1,-1)
            #agent = A2CAgent(state_size,action_size)

        # scores, episodes = [], []
        # for i in range(1):
            returns=deque(maxlen=1000)
            for e in range(EPISODES):
                done = False
                # count=0
                episodic_reward = 0
                full_state=[]

                observations, infos = env.reset()
                state0=observations.get('agent_0')
                full_state.append(state0)
                state1=observations.get('agent_1')
                full_state.append(state1)
                state2=observations.get('agent_2')
                full_state.append(state2)

                full_state = np.reshape(full_state,[1,no_agent*state_size])
                full_state=np.array(full_state)   
                # state_tensor = torch.FloatTensor(state)
                
                # epi_count = 0
                # store_count = 0
                
                
                #state = np.reshape(state, [1, state_size])
                
                # for time in range(episode_time):
                
                while (env.agents):
                    # count+=1
                    # print("episode time:",count)
                    #env.render()
                    next_full_state=[]
                    # reward_single_stage=0
                     # action = agent.get_action(state)
                    # action1,action2,action3 = intrepret_action(action)
                    log_prob0,action0, log_prob1,action1, log_prob2,action2 =agent.get_action(state0,state1,state2)
                    # actions = [action0,action1,action2]
                    # print("action list",actions)
                    action = {'agent_0': action0, 'agent_1': action1, 'agent_2': action2}
                    observations, rewards, terminations, truncations, infos = env.step(action)
                    nx_state0=observations.get('agent_0')
                    next_full_state.append(nx_state0)
                    nx_state1=observations.get('agent_1')
                    next_full_state.append(nx_state1)
                    nx_state2=observations.get('agent_2')
                    next_full_state.append(nx_state2)
                    
                    # next_state = observations
                    next_full_state=np.reshape(next_full_state,[1,no_agent*state_size])
                    next_full_state=np.array(next_full_state) 

                    reward_single_stage = -(rewards.get('agent_0')+rewards.get('agent_1')+rewards.get('agent_2'))/no_agent
                    if reward_single_stage > 1:
                        reward_single_stage= 1
                    # reward_single_stage= -reward_single_stage/3
                    episodic_reward+=reward_single_stage
                    
                    # print("Before train model")
                    # if terminations.get('agent_0') and terminations.get('agent_1') and terminations.get('agent_2'):
                    #     done=True 
                    # else:
                    #     done=False    
                    agent.train_model(full_state,log_prob0,log_prob1,log_prob2,reward_single_stage,next_full_state, hid_st,done)
                    
                    state0 = nx_state0
                    state1 = nx_state1
                    state2 = nx_state2
                    full_state=next_full_state
                            
                    # episodic_reward = episodic_reward + (0.9**count)*total_reward
                returns.append(episodic_reward)
                average_ep_reward=np.mean(returns)
                avg_reward.append(average_ep_reward)
                f.write("{}\t{}\n".format(e,round(average_ep_reward,4))) 
                # store_total_cost[0][e] = episodic_reward/(e+1)
                
                # pen_count = pen_count + epi_count
                
                # store_total_penalty[0][e] = pen_count/(e+1)
    
                # print("average reward",episodic_reward/(e+1))
                if(e%10==0):
                    print("episode is: {}, episodic cost: {}".format(e, episodic_reward))
                    hid_st=hid_st.detach()
                    full_state = torch.FloatTensor(full_state)
                    # print("shape........",state_tensor.shape)
                    state_tensor_b = torch.cat([full_state, hid_st], 1)
                    print("value:::",agent.critic(state_tensor_b)[0])
                    plot1 = plt.figure(1)
                    plt.plot(avg_reward,'r')
                    plt.xlabel("No. of Episodes")
                    plt.ylabel("Average Cost")
                    # plt.title('No. of l-training iterations vs Acrobot training Standard dev.')
                    plt.savefig("./plot_avg_cost_AC_MARL.svg")
                    plt.savefig("./plot_avg_cost_AC_MARL") 
                if(e%100==0):
                    torch.save(agent.actor0, actor0_save_path)
                    torch.save(agent.actor1, actor1_save_path)
                    torch.save(agent.actor2, actor2_save_path)
                    torch.save(agent.critic, critic_save_path)    



  
