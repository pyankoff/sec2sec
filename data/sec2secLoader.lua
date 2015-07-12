-- loader for sec2sec

require 'torch'
require 'math'
local generate_sums = require 'data.generate_sums'

local sec2secLoader = {}
sec2secLoader.__index = sec2secLoader

function sec2secLoader.create(nbatches, batch_size, seq_length)
    local self = {}
    setmetatable(self, sec2secLoader)

    -- construct a tensor with all the data
    print('generating data...')
    self.vocab_mapping = {['1']=1, ['2']=2, ['3']=3, ['4']=4, ['5']=5, 
        ['6']=6, ['7']=7, ['8']=8, ['9']=9, ['0']=10, ['+']=11, 
        ['=']=12, ['.']=13, ['-']=14, ['@']=15}

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    self.batch_size = batch_size
    self.seq_length = seq_length

    xdata = torch.Tensor(nbatches*batch_size*seq_length)
    ydata = torch.Tensor(nbatches*batch_size*seq_length)
    
    idx = 0
    while true do
        input, target, orig = unpack(generate_sums.generate())

        if idx + #input <= nbatches*batch_size*seq_length then
            maxi = #input
        else
            maxi = nbatches*batch_size*seq_length - idx
        end
        
        for i=1,maxi do
            xdata[idx+i] = input[i]
            ydata[idx+i] = target[i]
        end

        idx = idx + maxi
        
        if idx >= nbatches*batch_size*seq_length then
            break
        end
    end
    
    self.x_batches = {}
    self.y_batches = {}

    self.x_batches = xdata:view(batch_size, -1):split(seq_length, 2) 
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2) 
    
    assert(#self.x_batches == #self.y_batches)
    self.nbatches = #self.x_batches

    self.current_batch = 0
    self.evaluated_batches = 0  -- number of times next_batch() called

    print('data load done.')
    collectgarbage()
    return self
end

-- *** STATIC method ***
function sec2secLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('loading text file...')
    local f = torch.DiskFile(in_textfile)
    local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
    f:close()

    -- create vocabulary if it doesn't exist yet
    print('timer: ', timer:time().real)
    print('creating vocabulary mapping...')
    -- record all of them into a set
    local unordered = {}
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end

    -- sort them
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered) -- now order maps int->char

    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end

    -- construct a tensor with all the data
    print('timer: ', timer:time().real)
    print('putting data into tensor...')
    local data = torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
    end

    print('saving two files...')
    torch.save(out_vocabfile, vocab_mapping)
    torch.save(out_tensorfile, data)

    print('Done in time (seconds): ', timer:time().real)
end

function sec2secLoader:next_batch()
    self.current_batch = (self.current_batch % self.nbatches) + 1
    self.evaluated_batches = self.evaluated_batches + 1
    return self.x_batches[self.current_batch], self.y_batches[self.current_batch]
end

return sec2secLoader

