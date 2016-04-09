require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'image'

require 'modules.QAAttentionModel'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Demo image and questions
-------------------------------------------------------------------------------

-- You can replace it with your own image
local image_file = 'data/demo.jpg'

-- You can write your own questions here
local questions = {
  'How many people are there?',
  'What animal can be seen in the picture?',
  'Who is wearing a red shirt?',
  'Where color is the elephant?',
  'When is the picture taken?'
}

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image QA demo')
cmd:text()
cmd:text('Options')

cmd:option('-model', 'checkpoints/model_visual7w_telling_cpu.t7', 'path to model to evaluate')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1234, 'random number generator seed to use')

local opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Basic setup and load pretrianed model
-------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
gpu_mode = opt.gpuid >= 0

if gpu_mode then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

assert(string.len(opt.model) > 0, 'error: must provide a model')
local checkpoint = torch.load(opt.model)
local vocab = checkpoint.vocab
local vocab_size = 0
local word_to_ix = {}
for i, w in pairs(vocab) do
  word_to_ix[w] = i
  vocab_size = vocab_size + 1
end

local modules = checkpoint.modules
local cnn = modules.cnn
local rnn = modules.rnn
modules = nil
collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Prepare inputs
-------------------------------------------------------------------------------
local MAX_Q_LEN = 15
local num_q = #questions
local question_labels = torch.LongTensor(MAX_Q_LEN, num_q):zero()
local q_len = torch.LongTensor(num_q):zero()
for k, q in pairs(questions) do
  local s = string.lower(q):gsub('%p', '')
  for token in s:gmatch('%w+') do
    if q_len[k] < MAX_Q_LEN and word_to_ix[token] then
      q_len[k] = q_len[k] + 1
      question_labels[q_len[k]][k] = word_to_ix[token]
    end
  end
  q_len[k] = q_len[k] + 1
  question_labels[q_len[k]][k] = vocab_size
end

-------------------------------------------------------------------------------
-- Start demo
-------------------------------------------------------------------------------
-- forward CNN
local img = image.load(image_file)
img = image.scale(img, 224, 224, 'bicubic'):view(1, 3, 224, 224):mul(255)
img = net_utils.prepro(img, false, gpu_mode)

-- encode image and question tokens with a pretrained LSTM
if gpu_mode then cnn:cuda() end
local image_encodings = cnn:forward(img)

-- convolutional feature maps for attention
-- layer #30 in VGG outputs the 14x14 conv5 features
local conv_feat_maps = cnn:get(30).output:clone()
conv_feat_maps = conv_feat_maps:view(1, 512, -1)

image_encodings = torch.repeatTensor(image_encodings, num_q, 1)
conv_feat_maps = torch.repeatTensor(conv_feat_maps, num_q, 1, 1)

cnn = nil
collectgarbage() -- free some memory

-- forward RNN
rnn:createClones()
if gpu_mode then rnn:cuda() end

-- forward the model to also get generated samples for each image
local answer_labels = rnn:sample(image_encodings, conv_feat_maps, question_labels, q_len)

rnn = nil
collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Output results
-------------------------------------------------------------------------------
local questions = net_utils.decode_sequence(vocab, question_labels)
local answers = net_utils.decode_sequence(vocab, answer_labels)
print('** QA demo on ' .. image_file .. ' **\n')
for k = 1, #answers do
  print(string.format('Q: %s ?', questions[k]))
  print(string.format('A: %s .\n', answers[k]))
end
