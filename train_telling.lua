require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'
require 'loadcaffe'

-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.QADatasetLoader'
require 'modules.QAAttentionModel'
require 'modules.QACriterion'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5', 'data/qa_data.h5', 'path to the h5file containing the preprocessed dataset')
cmd:option('-input_json', 'data/qa_data.json', 'path to the json file containing additional info and vocab')
cmd:option('-dataset_file', 'visual7w-toolkit/datasets/visual7w-telling/dataset.json', 'path to the json file containing the dataset')
cmd:option('-vgg_proto', 'cnn_models/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to VGGNet-16 prototxt.')
cmd:option('-vgg_model', 'cnn_models/VGG_ILSVRC_16_layers.caffemodel', 'path to VGGNet-16 Caffe model.')
cmd:option('-load_model_from', '', 'path to a model checkpoint to initialize model weights from.')

-- Model settings
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size', 512, 'the encoding size of each token in the vocabulary and image fc7.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 64, 'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

-- Optimization: learning rate decay
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')

-- Optimization: for the RNN
cmd:option('-rnn_optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-rnn_learning_rate', 4e-4, 'learning rate')
cmd:option('-rnn_optim_alpha', 0.8, 'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-rnn_optim_beta', 0.999, 'beta used for adam')
cmd:option('-rnn_optim_epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim', 'adam', 'optimization to use for CNN')
cmd:option('-cnn_optim_alpha', 0.8, 'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta', 0.999, 'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-num_image_eval', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-mc_evaluation', false, 'whether to use multiple-choice metrics (false = disable)')
cmd:option('-verbose', false, 'verbose mode: output additional logs')

-- misc
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-suffix', '', 'suffix that is appended to the model file paths')
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
  cutorch.setDevice(opt.gpuid+1)
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = QADatasetLoader{h5_file = opt.input_h5, json_file = opt.input_json, dataset_file=opt.dataset_file}

-------------------------------------------------------------------------------
-- Load initial network
-------------------------------------------------------------------------------
local modules = {}

if string.len(opt.load_model_from) > 0 then
  -- load modules from file
  print('initializing weights from ' .. opt.load_model_from)
  local loaded_checkpoint = torch.load(opt.load_model_from)
  modules = loaded_checkpoint.modules
  net_utils.unsanitize_gradients(modules.cnn)
  local rnn_modules = modules.rnn:getModulesList()
  for k,v in pairs(rnn_modules) do net_utils.unsanitize_gradients(v) end
  modules.crit = nn.QACriterion() -- not in checkpoints, create manually
else
  -- create modules from scratch
  -- intialize attention QA model
  local rnn_opt = {}
  rnn_opt.vocab_size = loader:getVocabSize()
  rnn_opt.seq_length = loader:getSeqLength()
  rnn_opt.input_encoding_size = opt.input_encoding_size
  rnn_opt.rnn_size = opt.rnn_size
  rnn_opt.num_layers = 1
  rnn_opt.dropout = opt.drop_prob_lm
  rnn_opt.batch_size = opt.batch_size
  modules.rnn = nn.QAAttentionModel(rnn_opt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.vgg_proto, opt.vgg_model, cnn_backend)
  modules.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
  modules.crit = nn.QACriterion()
end

-- ship everything to GPU
if gpu_mode then
  for k,v in pairs(modules) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector
local rnn_params, rnn_grad_params = modules.rnn:getParameters()
local cnn_params, cnn_grad_params = modules.cnn:getParameters()
print('total number of parameters in RNN: ', rnn_params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_rnn = modules.rnn:clone()
thin_rnn.core:share(modules.rnn.core, 'weight', 'bias')
thin_rnn.lookup_table:share(modules.rnn.lookup_table, 'weight', 'bias')
thin_rnn.attention_nn:share(modules.rnn.attention_nn, 'weight', 'bias')
local rnn_modules = thin_rnn:getModulesList()
for k,v in pairs(rnn_modules) do net_utils.sanitize_gradients(v) end

local thin_cnn = modules.cnn:clone('weight', 'bias')
net_utils.sanitize_gradients(thin_cnn)
collectgarbage() -- free some memory

-- create clones and ensure parameter sharing. we have to do this
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
modules.rnn:createClones()

local iter = 0
collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  modules.cnn:training()
  modules.rnn:training()
  rnn_grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  local batch = loader:getBatch{split='train', batch_size=opt.batch_size}

  -- do data augmentation
  batch.images = net_utils.prepro(batch.images, false, gpu_mode)

  -- load image features
  local image_encodings = modules.cnn:forward(batch.images)

  -- convolutional feature maps for attention
  -- layer #30 in VGGNet-16 outputs the 14x14 512-dimensional conv5 features
  local conv_feat_maps = modules.cnn:get(30).output:view(opt.batch_size, 512, -1)
  local logprobs = modules.rnn:forward{image_encodings, conv_feat_maps, batch.labels}
  local loss = modules.crit:forward(logprobs, {batch.labels, batch.question_length})

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = modules.crit:backward(logprobs, {batch.labels, batch.question_length})
  -- backprop language model
  local dfc, dconv, _ = unpack(modules.rnn:backward({image_encodings, conv_feat_maps, batch.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dx = modules.cnn:backward(batch.images, dfc)
  end

  -- clip gradients
  rnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_image_eval = utils.getopt(evalopt, 'num_image_eval', -1)
  local shuffle = utils.getopt(evalopt, 'shuffle', (num_image_eval>=0))

  modules.cnn:evaluate()
  modules.rnn:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local ncorrect = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while num_image_eval < 0 or n < num_image_eval do

    -- get batch of data  
    local batch = loader:getBatch{split=split, batch_size=opt.batch_size, shuffle=shuffle}

    -- no data augmentation
    batch.images = net_utils.prepro(batch.images, false, gpu_mode)

    -- load image features
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

    -- forward the model to also get generated samples for each image
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
    else
      -- sample from image and question tokens
      local seq = modules.rnn:sample(image_encodings, conv_feat_maps, question_labels, q_len)
      local questions = net_utils.decode_sequence(vocab, question_labels)
      local answers = net_utils.decode_sequence(vocab, seq)
      for k = 1, #answers do
        local entry = {qa_id = batch.qa_id[k], question = questions[k], answer = answers[k]}
        table.insert(predictions, entry)
        if verbose then
          print(string.format('question %s: %s ? %s .', entry.qa_id, entry.question, entry.answer))
        end
      end
    end
    
    if verbose then
      print(string.format('evaluating validation performance... %d (%f)', n, loss))
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

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss_init
local loss_avg
local loss_history = {}
local val_accuracy_history = {}
local val_loss_history = {}
local best_score
-- optimization options
local rnn_optim_opt = {optim_alpha=opt.rnn_optim_alpha, optim_beta=opt.rnn_optim_beta, optim_epsilon=opt.rnn_optim_epsilon, optim_state={}}
local cnn_optim_opt = {optim_alpha=opt.cnn_optim_alpha, optim_beta=opt.cnn_optim_beta, optim_epsilon=opt.cnn_optim_epsilon, optim_state={}}
while opt.max_iters < 0 or iter <= opt.max_iters do

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  if loss_avg == nil then loss_avg = losses.total_loss else loss_avg = 0.995*loss_avg+0.005*losses.total_loss end
  print(string.format('iter %d: %f (%f)', iter, losses.total_loss, loss_avg))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, val_accuracy = eval_split('val', {num_image_eval=opt.num_image_eval, verbose=opt.verbose})
    print('validation loss: ', val_loss)

    local current_score = - val_loss
    val_loss_history[iter] = val_loss

    -- record accuracy if doing multiple-choice evaluation
    if opt.mc_evaluation then
      current_score = val_accuracy
      val_accuracy_history[iter] = val_accuracy
      print('validation accuracy: ', val_accuracy)
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.suffix)

    -- write a json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions
    checkpoint.val_accuracy_history = val_accuracy_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the modules (which have weights) and save to file
        local save_modules = {}
        save_modules.rnn = thin_rnn
        save_modules.cnn = thin_cnn
        checkpoint.modules = save_modules
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both RNN and CNN
  local rnn_learning_rate = opt.rnn_learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    rnn_learning_rate = rnn_learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a RNN update
  grad_update(rnn_params, rnn_grad_params, rnn_learning_rate, opt.rnn_optim, rnn_optim_opt)

  -- do a CNN update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    grad_update(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim, cnn_optim_opt)
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end
  if loss_init == nil then loss_init = losses.total_loss end
  if losses.total_loss > loss_init * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end

end
