local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read("*all")
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

-- split the labels into two vectors of question and answer 
function utils.split_question_answer(labels, question_length, answer_length, question_seq_length, answer_seq_length)
  local batch_size = labels:size()[2]
  local q_length = question_seq_length or question_length:max()
  local a_length = answer_seq_length or answer_length:max()
  local question_labels = torch.LongTensor(q_length, batch_size)
  local answer_labels = torch.LongTensor(a_length, batch_size)
  question_labels:zero()
  answer_labels:zero()
  for i = 1, batch_size do
    local lq = question_length[i]
    local la = answer_length[i]
    question_labels[{{1,lq}, {i,i}}] = labels[{{1,lq}, {i,i}}]
    answer_labels[{{1,la}, {i,i}}] = labels[{{1+lq,lq+la}, {i,i}}]
  end
  return question_labels, answer_labels
end

return utils
