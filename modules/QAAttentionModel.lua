require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'modules.LSTM'

-------------------------------------------------------------------------------
-- Spatial Attention QA model
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.QAAttentionModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size', 512)
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.seq_length = utils.getopt(opt, 'seq_length')

  -- options for attention
  self.conv_feat_num = utils.getopt(opt, 'conv_feat_num', 196)
  self.conv_feat_size = utils.getopt(opt, 'conv_feat_size', 512)
  self.attention_size = utils.getopt(opt, 'attention_size', 256)

  -- create the core lstm network. note +1 for both the START and END tokens
  local dropout = utils.getopt(opt, 'dropout', 0)
  self.lstm_input_size = self.input_encoding_size + self.attention_size
  self.core = LSTM.lstm(self.lstm_input_size, self.vocab_size+1, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)

  -- create attention network
  local att_opt = {}
  att_opt.rnn_size = self.rnn_size
  att_opt.conv_feat_size = self.conv_feat_size
  att_opt.conv_feat_num = self.conv_feat_num
  att_opt.attention_size = self.attention_size
  self.attention_nn = build_attention_nn(att_opt)

  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the QA model')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  self.attention_nns = {self.attention_nn}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
    self.attention_nns[t] = self.attention_nn:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table, self.attention_nn}
end

function layer:parameters()
  -- flatten model parameters and gradients into single vectors
  local params, grad_params = {}, {}
  for k, m in pairs(self:getModulesList()) do
    local p, g = m:parameters()
    for _, v in pairs(p) do table.insert(params, v) end
    for _, v in pairs(g) do table.insert(grad_params, v) end
  end
  -- invalidate clones as weight sharing breaks
  self.clones = nil
  -- return all parameters and gradients
  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.attention_nns) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.attention_nns) do v:evaluate() end
end

--[[
takes an image and question, run the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(fc_feats, conv_feats, question_labels, question_lengths, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  local batch_size = fc_feats:size(1)

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  for i = 1, batch_size do
    -- initialize state
    self:_createInitState(1)
    local state = self.init_state

    local fc_feat = fc_feats[{{i,i}, {}}]
    local conv_feat = conv_feats[{{i,i}, {}, {}}]
    local input_tokens = question_labels[{{}, {i,i}}]
    local offset = question_lengths[i]

    for t=1,self.seq_length+2 do

      local xt, it, sampleLogprobs
      if t == 1 then
        -- feed in the images
        xt = fc_feat
      elseif t == 2 then
        -- feed in the start tokens
        it = torch.LongTensor({self.vocab_size+1})
        xt = self.lookup_table:forward(it)
      elseif t <= 2 + offset then
        -- feed in the question tokens
        it = torch.LongTensor(input_tokens[t-2])
        xt = self.lookup_table:forward(it)
      else
        -- take predictions from previous time step and feed them in
        if sample_max == 1 then
          -- use argmax "sampling"
          sampleLogprobs, it = torch.max(logprobs, 2)
          it = it:view(-1):long()
        else
          -- sample from the distribution of previous predictions
          local prob_prev
          if temperature == 1.0 then
            prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
          else
            -- scale logprobs by temperature
            prob_prev = torch.exp(torch.div(logprobs, temperature))
          end
          it = torch.multinomial(prob_prev, 1)
          sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
          it = it:view(-1):long() -- and flatten indices for downstream processing
        end
        xt = self.lookup_table:forward(it)
      end

      local k = t-2-offset
      if k >= 1 then -- starting answer sequence
        seq[k][i] = it -- record the samples
        seqLogprobs[k][i] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
      end

      -- get attention feature
      local h_state = state[self.num_state]
      local att = self.attention_nn:forward({conv_feat, h_state})

      -- construct the inputs
      xt = torch.cat(xt, att)
      local inputs = {xt,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for j=1,self.num_state do table.insert(state, out[j]) end
    end

  end
  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local fc_feats = input[1]
  local conv_feats = input[2]
  local seq = input[3]

  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt
    if t == 1 then
      -- feed in the images
      xt = fc_feats -- NxK sized input
    elseif t == 2 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1
      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

    if not can_skip then
      -- get attention feature
      local h_state = self.state[t-1][self.num_state]
      local att = self.attention_nns[t]:forward({conv_feats, h_state})
      -- construct the inputs
      xt = torch.cat(xt, att)
      self.inputs[t] = {xt,unpack(self.state[t-1])}
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end

-- compute backprop gradients 
-- gradOutput is an (D+2)xNx(M+1) Tensor.
function layer:updateGradInput(input, gradOutput)
  local conv_feats = input[2]
  local dimgs -- grad on input images
  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  local dconv_feats = conv_feats:clone():zero()
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end

    -- backprop image and vocab lookup
    local dwt = dxt[{{},{1,self.input_encoding_size}}]
    local dat = dxt[{{},{self.input_encoding_size+1,self.lstm_input_size}}]

    -- backprop attention net
    local h_state = self.state[t-1][self.num_state]
    local datt = self.attention_nns[t]:backward({conv_feats, h_state}, dat)
    local dconv, dh_state = unpack(datt)

    -- accumulate grads for conv maps
    dconv_feats:add(dconv)

    -- update grads for previous h_state
    dstate[t-1][self.num_state]:add(dh_state)

    -- continue backprop of xt
    if t == 1 then
      dimgs = dwt
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dwt) -- backprop into lookup table
    end
  end
  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, dconv_feats, torch.Tensor()}
  return self.gradInput
end

-- create a NN graph module for the attention net
-- attention net takes two inputs, previous LSTM hidden state and convolutional feature map
-- it produces a weighted convolutional feature using soft attention
function build_attention_nn(opt)
  local conv_feat_maps = nn.Identity()()
  local prev_h = nn.Identity()()
  -- compute attention coefficients
  local flatten_conv = nn.View(-1):setNumInputDims(2)(conv_feat_maps)
  local f_conv = nn.Linear(opt.conv_feat_size*opt.conv_feat_num, opt.conv_feat_num)(flatten_conv)
  local f_h = nn.Linear(opt.rnn_size, opt.conv_feat_num)(prev_h)
  local f_sum = nn.Tanh()(nn.CAddTable()({f_conv, f_h}))
  local coef = nn.SoftMax()(f_sum)
  local coef_expanded = nn.Reshape(opt.conv_feat_num, 1)(coef)
  -- compute soft spatial attention
  local soft_att = nn.MM()({conv_feat_maps, coef_expanded})
  local att_conv = nn.View(-1):setNumInputDims(2)(soft_att)
  local att_out = nn.ReLU()(nn.Linear(opt.conv_feat_size, opt.attention_size)(att_conv))
  -- create nn graph module
  return nn.gModule({conv_feat_maps, prev_h}, {att_out})
end