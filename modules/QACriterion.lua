require 'nn'

-------------------------------------------------------------------------------
-- Sequence-to-Sequence QA Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.QACriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

function crit:updateOutput(input, labels)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local seq, encoder_len = unpack(labels)
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local loss = 0
  local loss_per_sample = torch.zeros(N)
  local n_total = 0
  for b = 1, N do -- iterate over batches
    local first_time = true
    -- iterate over sequence time
    -- ignore t=1, dummy forward for the image and t=2..encoder_len+1 (question tokens)
    local n_token = 0
    for t = encoder_len[b] + 2, L do
      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t-1,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end
      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        local logp = input[{ t,b,target_index }]
        loss = loss - logp -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n_total = n_total + 1
        -- record loss of this sample
        loss_per_sample[b] = loss_per_sample[b] - logp
        n_token = n_token + 1
      end
    end
    -- normalize by the number of tokens in the answer
    loss_per_sample[b] = loss_per_sample[b] / n_token
  end
  self.loss_per_sample = loss_per_sample
  -- normalize by number of predictions that were made
  self.output = loss / n_total
  self.gradInput:div(n_total)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
