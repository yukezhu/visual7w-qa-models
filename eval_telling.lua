require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'

local utils = require 'misc.utils'
require 'misc.QADatasetLoader'
require 'modules.QAAttentionModel'
require 'modules.QACriterion'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate an Image QA model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
cmd:option('-input_h5','data/qa_data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/qa_data.json','path to the json file containing additional info and vocab')
cmd:option('-dataset_file','visual7w-toolkit/datasets/visual7w-telling/dataset.json','path to the json file containing the dataset')

-- Basic options
cmd:option('-split', 'test', 'split to use: val|test|train')
cmd:option('-mc_evaluation', false, 'whether to use multiple-choice metrics (false = disable)')
cmd:option('-batch_size', 16, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_image_eval', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-verbose', false, 'verbose mode: output additional logs')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

local opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
gpu_mode = opt.gpuid >= 0

if gpu_mode then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'error: must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = QADatasetLoader{h5_file = opt.input_h5, json_file = opt.input_json, dataset_file=opt.dataset_file}

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local modules = checkpoint.modules
modules.crit = nn.QACriterion()
modules.rnn:createClones() -- reconstruct clones inside the language model
if gpu_mode then for k,v in pairs(modules) do v:cuda() end end

collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_image_eval = utils.getopt(evalopt, 'num_image_eval', -1)

  modules.rnn:evaluate()
  modules.cnn:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local ncorrect = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  while num_image_eval < 0 or n < num_image_eval do

    -- get batch of data  
    local batch = loader:getBatch{split=split, batch_size=opt.batch_size, shuffle=false}

    -- no data augmentation
    batch.images = net_utils.prepro(batch.images, false, gpu_mode)

    -- encode image and question tokens with a pretrained LSTM
    local image_encodings = modules.cnn:forward(batch.images)

    -- convolutional feature maps for attention
    -- layer #30 in VGG outputs the 14x14 conv5 features
    local conv_feat_maps = modules.cnn:get(30).output:clone()
    conv_feat_maps = conv_feat_maps:view(opt.batch_size, 512, -1)
    local logprobs = modules.rnn:forward{image_encodings, conv_feat_maps, batch.labels}
    local loss = modules.crit:forward(logprobs, {batch.labels, batch.question_length})

    n = n + #batch.infos
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    local q_len, a_len = batch.question_length, batch.answer_length
    local question_labels, answer_labels = utils.split_question_answer(batch.labels, q_len, a_len)

    -- use multiple choices for evaluation
    if opt.mc_evaluation then
      local questions = net_utils.decode_sequence(vocab, question_labels)
      for j = 1, #batch.infos do
        -- prepare data
        local mc_labels = loader:getMultipleChoices(batch.qa_id[j])
        local mc_length = mc_labels:size(1)
        local noption = mc_labels:size(2) + 1
        local offset = batch.question_length[j]
        local labels = torch.repeatTensor(batch.labels[{{}, {j,j}}], 1, noption)
        labels[{{offset+1, offset+mc_length}, {2, noption}}] = mc_labels
        question_length = torch.LongTensor(noption):fill(offset)
        -- forward the multiple choices in a batch
        local mc_image_encodings = torch.repeatTensor(image_encodings[j], noption, 1)
        local mc_conv_feat_maps = torch.repeatTensor(conv_feat_maps[j], noption, 1, 1)
        local logprobs = modules.rnn:forward{mc_image_encodings, mc_conv_feat_maps, labels}
        -- compute loss for each of the multiple choice option
        modules.crit:forward(logprobs, {labels, question_length})
        local mc_loss = modules.crit.loss_per_sample
        -- select the option with the smallest loss as the answer
        local _, selected = mc_loss:min(1)
        local k = selected[1]
        if k == 1 then ncorrect = ncorrect + 1 end
        -- record predictions
        local answers = net_utils.decode_sequence(vocab, labels[{{offset+1, offset+mc_length}}])
        local entry = {qa_id = batch.qa_id[j], question = questions[j], answer = answers[k], selected=k}
        table.insert(predictions, entry)
        if verbose then
          print(string.format('question %s: %s ? %s .', entry.qa_id, entry.question, entry.answer))
        end
      end
    else -- use freeform evaluation
      -- evaluate loss if we have the labels
      if batch.labels then
        local logprobs = modules.rnn:forward{image_encodings, conv_feat_maps, batch.labels}
        loss = modules.crit:forward(logprobs, {batch.labels, batch.question_length})
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1
      end
      -- forward the model to also get generated samples for each image
      local seq = modules.rnn:sample(image_encodings, conv_feat_maps, question_labels, q_len)
      local questions = net_utils.decode_sequence(vocab, question_labels)
      local answers = net_utils.decode_sequence(vocab, seq)
      for k=1,#answers do
        local entry = {qa_id = batch.qa_id[k], question = questions[k], answer = answers[k]}
        table.insert(predictions, entry)
        if verbose then
          print(string.format('question %s: %s ? %s .', entry.qa_id, entry.question, entry.answer))
        end
      end
    end

    if verbose then
      print(string.format('evaluating %s performance... %d (%f)', split, n, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if batch.bounds.wrapped then break end -- the split ran out of data, lets break out
  end

  local accuracy
  if opt.mc_evaluation then
    accuracy = ncorrect / n
  end

  return loss_sum/loss_evals, predictions, accuracy
end

local loss, split_predictions, acc = eval_split(opt.split, {num_image_eval = opt.num_image_eval, verbose=opt.verbose})
print('loss: ', loss)
print('acc: ', acc)

-- dump the json
json_struct = {split_predictions = split_predictions, loss = loss, acc = acc}
utils.write_json(string.format('results/results_%s.json', opt.split), json_struct)
