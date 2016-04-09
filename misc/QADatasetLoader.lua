require 'hdf5'
local utils = require 'misc.utils'

local QADatasetLoader = torch.class('QADatasetLoader')

function QADatasetLoader:__init(opt)

  -- load the json file which contains the dataset information
  print('QADatasetLoader loading dataset file: ', opt.dataset_file)
  self.dataset = utils.read_json(opt.dataset_file)
  self.nimage = #self.dataset.images
  self.num_mc = utils.getopt(opt, 'num_mc', 3) -- num. of multiple-choices per question
  print('image size is ' .. self.nimage)

  -- load the json file which contains ix_to_word mapping
  print('QADatasetLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)

  -- append a START token to the end of the vocabulary list
  self.vocab_size = self.vocab_size + 1
  self.ix_to_word[tostring(self.vocab_size)] = '' -- START token for generating answers
  print('vocab size is ' .. self.vocab_size)

  -- open the hdf5 file
  print('QADatasetLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')

  -- load in the sequence data
  local seq_size
  seq_size = self.h5_file:read('question_label'):dataspaceSize()
  self.question_seq_length = seq_size[2]
  print('max question sequence length in data is ' .. self.question_seq_length)

  seq_size = self.h5_file:read('answer_label'):dataspaceSize()
  self.answer_seq_length = seq_size[2]
  print('max answer sequence length in data is ' .. self.answer_seq_length)
  self.seq_length = self.question_seq_length + self.answer_seq_length

  -- load the pointers in full to memory (should be small enough)
  self.qa_id = self.h5_file:read('qa_id'):all()
  self.qa_start_ix = self.h5_file:read('qa_start_ix'):all()
  self.qa_end_ix = self.h5_file:read('qa_end_ix'):all()

  -- map image id to index
  self.image_id_to_idx = {}
  for k = 1, #self.dataset.images do
    self.image_id_to_idx[self.dataset.images[k].image_id] = k
  end

  -- map image index in hdf5 file
  local image_id = self.h5_file:read('image_id'):all()
  self.image_data = self.h5_file:read('image_data')
  self.image_id_to_h5idx = {}
  for k = 1, image_id:nElement() do
    self.image_id_to_h5idx[image_id[k]] = k
  end

  -- map QA id to index
  self.qa_id_to_idx = {}
  for k = 1, self.qa_id:nElement() do
    self.qa_id_to_idx[self.qa_id[k]] = k
  end

  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  self.split_qa_ix = {}
  self.qa_iterators = {}
  for i,img in pairs(self.dataset.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    if not self.split_qa_ix[split] then
      self.split_qa_ix[split] = {}
      self.qa_iterators[split] = 1
    end
    for j,qa in pairs(img.qa_pairs) do
      table.insert(self.split_qa_ix[split], {img.image_id, qa.qa_id})
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end
end

function QADatasetLoader:resetIterator(split)
  self.iterators[split] = 1
end

function QADatasetLoader:getVocabSize()
  return self.vocab_size
end

function QADatasetLoader:getImageSize()
  return self.nimage
end

function QADatasetLoader:getMultipleChoiceSize()
  return self.num_mc
end

function QADatasetLoader:getVocab()
  return self.ix_to_word
end

function QADatasetLoader:getQuestionSeqLength()
  return self.question_seq_length
end

function QADatasetLoader:getAnswerSeqLength()
  return self.answer_seq_length
end

function QADatasetLoader:getSeqLength()
  return self.seq_length
end

function QADatasetLoader:getQuestionIndex(id)
  return self.qa_id_to_idx[id]
end

function QADatasetLoader:getImageIndex(id)
  return self.image_id_to_idx[id]
end

function QADatasetLoader:getMultipleChoices(id)
  local i = self:getQuestionIndex(id)
  local mc_labels = self.h5_file:read('mc_label'):partial(i, {1,self.num_mc}, {1,self.answer_seq_length})
  return mc_labels[1]:long():t()
end

function QADatasetLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 16) -- how many images get returned at one time (to go through CNN)
  local shuffle = utils.getopt(opt, 'shuffle', true) -- true, random sample (for train); false, sequential iteration (for eval)
  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local label_batch = torch.LongTensor(batch_size, self.seq_length)
  local qa_id_batch = torch.LongTensor(batch_size)
  local question_length_batch = torch.LongTensor(batch_size)
  local answer_length_batch = torch.LongTensor(batch_size)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  
  local split_qa_ix = self.split_qa_ix[split]
  local max_qa_index = #split_qa_ix
  
  for i = 1, batch_size do

    local question_seq, answer_seq
    local seq, id_seq
    local ixl
    if shuffle then
      local ri = self.iterators[split] -- get next index from iterator
      local ri_next = ri + 1 -- increment iterator
      if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
      self.iterators[split] = ri_next
      ix = split_ix[ri]
      assert(ix ~= nil, 'error: split ' .. split .. ' was accessed out of bounds with ' .. ri)
      -- fetch the sequence labels
      local ix1 = self.qa_start_ix[ix]
      local ix2 = self.qa_end_ix[ix]
      local nseq = ix2 - ix1 + 1 -- number of QA pairs available for this image
      assert(nseq > 0, 'error: an image does not have any QA')
      -- there is enough data to read a contiguous chunk, but subsample the chunk position
      ixl = torch.random(ix1, ix2) -- generates integer in the range
    else
      local ri = self.qa_iterators[split] -- get next index from iterator
      local ri_next = ri + 1 -- increment iterator
      if ri_next > max_qa_index then ri_next = 1; wrapped = true end
      self.qa_iterators[split] = ri_next
      ix, ixl = unpack(split_qa_ix[ri])
      ix = self:getImageIndex(ix)
      ixl = self:getQuestionIndex(ixl)
    end
    
    local question_length = self.h5_file:read('question_label_length'):partial({ixl, ixl})
    local answer_length = self.h5_file:read('answer_label_length'):partial({ixl, ixl})
    question_length = math.min(question_length[1], self.question_seq_length-1) -- leave one slot for START
    answer_length = math.min(answer_length[1], self.answer_seq_length)
    id_seq = self.h5_file:read('qa_id'):partial({ixl, ixl})

    -- copy question and answer sequences into a single vector
    seq = torch.LongTensor(self.seq_length)
    seq:zero() -- important to initialize with zeros
    if question_length > 0 then
      question_seq = self.h5_file:read('question_label'):partial({ixl, ixl}, {1,question_length})
      seq[{{1,question_length}}] = question_seq
      seq[question_length+1] = self.vocab_size -- last token: START
    end
    if answer_length > 0 then
      answer_seq = self.h5_file:read('answer_label'):partial({ixl, ixl}, {1,answer_length})
      seq[{{question_length+2, question_length+answer_length+1}}] = answer_seq
    end
    
    -- record question and answer lengths
    question_length_batch[i] = question_length + 1
    answer_length_batch[i] = answer_length

    -- store the question and answers in the batch tensor
    label_batch[i] = seq
    qa_id_batch[i] = id_seq

    -- and record associated info as well
    local info_struct = {}
    info_struct.image_id = self.dataset.images[ix].image_id
    info_struct.filename = self.dataset.images[ix].filename
    table.insert(infos, info_struct)
  end

  local data = {}
  data.labels = label_batch:transpose(1,2):contiguous()
  data.question_length = question_length_batch:contiguous()
  data.answer_length = answer_length_batch:contiguous()
  data.qa_id = qa_id_batch:contiguous()
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos

  -- read image from hdf5 file
  data.images = torch.Tensor(batch_size, 3, 224, 224)
  for i = 1, batch_size do
    local idx = self.image_id_to_h5idx[infos[i].image_id]
    data.images[i] = self.image_data:partial({idx,idx}, {1,3}, {1,224}, {1,224})
  end

  -- raw image pixel range [0, 1]
  data.images:mul(255)

  return data
end
