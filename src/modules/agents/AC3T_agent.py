import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import copy
import math

# from modules.estimators import REGISTRY as est_REGISTRY
from modules.state_encoders import REGISTRY as state_enc_REGISTRY
from types import SimpleNamespace as SN


class AC3TAgent(nn.Module):
    """
        VAE State Estimation Agent
        Each agent make decision based on estimated z and its observation
    """
    def __init__(self, input_shape, args):
        super(AC3TAgent, self).__init__()
        self.args = args
        
        self.prob=args.prob
        
        self.raw_input_shape = self._get_input_shape(input_shape)

        # define state estimator (observation integration function)
        state_dim = int(np.prod(args.state_shape))
        
        
        # args.state_repre_dim=int(args.state_repre_dim*(1-self.prob))
        
        print('args.state_repre_dim',args.state_repre_dim)
        
        self.encoder = state_enc_REGISTRY[args.state_encoder](input_shape=input_shape, output_shape=state_dim, n_agents=args.n_agents, latent_dim=args.state_repre_dim, args=args)

        # get some dimension information
        self.latent_dim = args.state_repre_dim * args.n_agents

        # network architecture relating to spr
        # z is in form of [n_agents, state_repr]

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.args.spr_dim),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(self.args.spr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.args.spr_dim),
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projection = copy.deepcopy(self.projection)
        for param in (list(self.target_encoder.parameters())
                    + list(self.target_projection.parameters())):
            param.requires_grad = False


        # network architecture relating to policy/rl
        # define some networks
        self.gate = nn.Linear(input_shape, self.latent_dim)
        

        self.ob_fc = nn.Linear(self.raw_input_shape, args.ob_embed_dim)
        self.fc1 = nn.Linear(args.ob_embed_dim + self.latent_dim + args.n_agents, args.hidden_dim)
   
        
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def encoder_init_hidden(self):
        # make hidden_states on same device as model
        return self.fc1.weight.new(1, self.args.encoder_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, encoder_hidden_state):
        
        
        # inputs.shape: [batch_size*n_agents, input_shape]
        bs = inputs.shape[0] // self.args.n_agents

        # decompose inputs
        raw_inputs, extra_inputs = self._build_inputs(inputs)

        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        elif "ae" in self.args.state_encoder:
            # z.shape: [batch_size, state_repr_dim]
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)

        else:
            raise ValueError("Unknown encoder!!!")
        
        # through gate
        weighted = F.sigmoid(self.gate(inputs)) # [bs*n_agents, state_repre_dim]
        if self.args.noise_env and self.args.noise_type == 1:
            repeated_z = z
        else:
            repeated_z = z.unsqueeze(1).repeat(1, self.args.n_agents, 1).reshape(bs*self.args.n_agents, -1) # [bs*n_agents, state_repre_dim]
        weighted_z = weighted * repeated_z 

        ob_embed = self.ob_fc(raw_inputs)   # [bs*n_agents, ob_embed_dim]
        
        #信道访问受限
        #print('ob_embed',ob_embed.shape)
        if self.prob>0:
            ob_embed=ob_embed.reshape(-1,self.args.n_agents,ob_embed.shape[-1])# [bs,n_agents, ob_embed_dim]
            del_k = math.floor(self.args.prob * self.args.n_agents)  # 删除多少消息*智能体总数得到删除几个智能体的消息
            # 计算第2和第3维度的绝对值之和
            sums = th.abs(ob_embed).sum(dim=2)
            # 找到绝对值之和最小的索引
            min_indices = th.argsort(sums)[:,:del_k]
            #print('min_indices',min_indices)
            
            # 创建一个新的张量来存储结果，避免原地修改
            new_msg_v = ob_embed.clone()
            # 将第1维度中索引为min_index的元素置为0
            new_msg_v[0, min_indices, :] = 0
            ob_embed=new_msg_v.reshape(-1,ob_embed.shape[-1])
            #print('ob_embed',ob_embed)


        action_inputs = th.cat([ob_embed, weighted_z, extra_inputs], dim=-1)


        # go through remaining networks
        x = F.relu(self.fc1(action_inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        return q, h, encoder_h

    def enc_forward(self, inputs, encoder_hidden_state):
        # inputs.shape: [bs*n_agents, input_shape]
        bs = inputs.shape[0] // self.args.n_agents

        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        elif "ae" in self.args.state_encoder:
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        else:
            raise ValueError("Unknown encoder!!!")

        return z, encoder_h
        
    def vae_forward(self, inputs, encoder_hidden_state):
        # inputs.shape: [bs*n_agents, input_shape]
        bs = inputs.shape[0] // self.args.n_agents
        if self.args.noise_env:
            noise = th.randn(bs*self.args.n_agents, inputs.shape[-1])
            inputs += noise.to(inputs.device)
        return self.encoder(inputs, encoder_hidden_state)
    
    def rl_forward(self, inputs, state_repr, hidden_state):
        # inputs.shape: [batch_size*n_agents, input_shape]
        bs = inputs.shape[0] // self.args.n_agents

        # decompose inputs
        raw_inputs, extra_inputs = self._build_inputs(inputs)

        # through gate
        weighted = F.sigmoid(self.gate(inputs)) # [bs*n_agents, state_repre_dim]
        repeated_z = state_repr.unsqueeze(1).repeat(1, self.args.n_agents, 1).reshape(bs*self.args.n_agents, -1) # [bs*n_agents, state_repre_dim]
        weighted_z = weighted * repeated_z 

        ob_embed = self.ob_fc(raw_inputs)   # [bs*n_agents, ob_embed_dim]


        action_inputs = th.cat([ob_embed, weighted_z, extra_inputs], dim=-1)


        # go through remaining networks
        x = F.relu(self.fc1(action_inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        return q, h

    def online_transform(self, inputs, encoder_hidden_state):
        """Compute the prediction of model learning loss.
        """
        # inputs.shape: [batch_size*n_agents, input_shape]
        # encoder_hidden_state: [batch_size, n_agents, encoder_hidden_dim]
        bs = inputs.shape[0] // self.args.n_agents

        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        elif "ae" in self.args.state_encoder:
            if self.args.noise_env:
                noise = th.randn(bs*self.args.n_agents, inputs.shape[-1])
                inputs += noise.to(inputs.device)
            # z.shape: [batch_size, self.latent_dim]
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        else:
            raise ValueError("Unknown encoder!!!")
        
        # do projection
        projected = self.projection(z)  # [batch_size, spr_dim]
        predicted = self.final_classifier(projected)    # [batch_size, spr_dim]

        return predicted, encoder_h

    def online_projection(self, z):
        """Compute the prediction of model learning loss when we already have z.
        """
        # z.shape: [batch_size, N, self.latent_dim]
        projected = self.projection(z)
        predicted = self.final_classifier(projected)
        return predicted
    
    def target_transform(self, inputs, encoder_hidden_state):
        """Compute the target of model learning loss.
        """
        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        elif "ae" in self.args.state_encoder:
            # z.shape: [batch_size, n_agents, state_repre_dim]
            z, encoder_h = self.target_encoder.encode(inputs, encoder_hidden_state)
        else:
            raise ValueError("Unknown encoder!!!")

        # do projection
        target_projected = self.target_projection(z)

        return target_projected, encoder_h

    def momentum_update(self):
        """Do momentum update for target encoder and target projection function
        """
        # Define update function for each net module
        def update_state_dict(model, state_dict, tau=1):
            """Update the state dict of ``model`` using the input ``state_dict``, which
            must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
            applies soft update: ``tau * new + (1 - tau) * old``.
            """
            if tau == 1:
                model.load_state_dict(state_dict)
            elif tau > 0:
                update_sd = {k: tau * state_dict[k] + (1 - tau) * v
                    for k, v in model.state_dict().items()}
                model.load_state_dict(update_sd)
        update_state_dict(self.target_encoder, self.encoder.state_dict(), self.args.momentum_tau)
        update_state_dict(self.target_projection, self.projection.state_dict(), self.args.momentum_tau)

    def enc_parameters(self):
        assert 0, "Shouldn't be called in current version of code."
        return self.encoder.parameters() 

    def rl_parameters(self):
        assert 0, "Shouldn't be called in current version of code."
        return list(self.gate.parameters()) + \
            list(self.ob_fc.parameters()) + \
            list(self.fc1.parameters()) + \
            list(self.rnn.parameters()) + \
            list(self.fc2.parameters())
    
    def _build_inputs(self, inputs):
        # extract different input components
        base_inputs = inputs[:, :self.raw_input_shape]
        extra_inputs = inputs[:, self.raw_input_shape:]
        return base_inputs, extra_inputs

    def _get_input_shape(self, input_shape):
        """get raw env obs shape"""
        if self.args.obs_last_action:
            input_shape -= self.args.n_actions
        if self.args.obs_agent_id:
            input_shape -= self.args.n_agents
        return input_shape


if __name__ == "__main__":
    args_config = {
        "n_agents": 5,
        "estimator": "mlp",
        "state_repre_dim": 32,
        "hidden_dim": 32,
        "use_rnn": False,
        "n_actions": 5,
    }
    