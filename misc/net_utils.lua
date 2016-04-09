local utils = require 'misc.utils'
local net_utils = {}

-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size))
  cnn_part:add(backend.ReLU(true))
  return cnn_part
end

-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end

function net_utils.listModules(net)
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
    end
  end
end

function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if word ~= '' then
        if j >= 2 then txt = txt .. ' ' end
        txt = txt .. word
      end
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

return net_utils
