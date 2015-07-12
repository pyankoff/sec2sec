require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local sec2secLoader = require 'data.sec2secLoader'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
require 'Embedding'                     -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-model','','contains just the protos table, and nothing else')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',30,'number of timesteps to unroll to')
cmd:option('-nbatches',100000,'number of batches to generate')
cmd:option('-rnn_size',400,'size of LSTM internal state')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {model=true, save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

local loader = sec2secLoader.create(opt.nbatches, opt.batch_size, opt.seq_length)
local vocab_size = loader.vocab_size  -- the number of distinct characters

-- define model prototypes for ONE timestep, then clone them
--
local protos = {}
if opt.model == '' then
    protos.embed = Embedding(vocab_size, opt.rnn_size)
    -- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
    protos.lstm1 = LSTM.lstm(opt)
    protos.lstm2 = LSTM.lstm(opt)
    protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
    protos.criterion = nn.ClassNLLCriterion()
else
    protos = torch.load(opt.model)
end

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.embed, 
                    protos.lstm1, protos.lstm2, protos.softmax)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate1_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate1_h = initstate1_c:clone()

local initstate2_c = initstate1_c:clone()
local initstate2_h = initstate1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate1_c = initstate1_c:clone()
local dfinalstate1_h = initstate1_c:clone()

local dfinalstate2_c = initstate1_c:clone()
local dfinalstate2_h = initstate1_c:clone()

-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm1_c = {[0]=initstate1_c} -- internal cell states of LSTM
    local lstm1_h = {[0]=initstate1_h} -- output values of LSTM
    local lstm2_c = {[0]=initstate2_c} -- internal cell states of LSTM
    local lstm2_h = {[0]=initstate2_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm1_c[t], lstm1_h[t] = unpack(clones.lstm1[t]:forward{embeddings[t], lstm1_c[t-1], lstm1_h[t-1]})
        lstm2_c[t], lstm2_h[t] = unpack(clones.lstm2[t]:forward{lstm1_h[t], lstm2_c[t-1], lstm2_h[t-1]})

        predictions[t] = clones.softmax[t]:forward(lstm2_h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm1_c = {[opt.seq_length]=dfinalstate1_c}  -- internal cell states of LSTM
    local dlstm1_h = {}                                 -- output values of LSTM
    local dlstm2_c = {[opt.seq_length]=dfinalstate2_c}
    local dlstm2_h = {} 

    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the softmax and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == opt.seq_length then
            assert(dlstm2_h[t] == nil)
            dlstm2_h[t] = clones.softmax[t]:backward(lstm2_h[t], doutput_t)

            dlstm1_h[t], dlstm2_c[t-1], dlstm2_h[t-1] = unpack(clones.lstm2[t]:backward(
                {lstm1_h[t], lstm2_c[t-1], lstm2_h[t-1]},
                {dlstm2_c[t], dlstm2_h[t]}
            ))
        else
            dlstm2_h[t]:add(clones.softmax[t]:backward(lstm2_h[t], doutput_t))

            dlstm1ht, dlstm2_c[t-1], dlstm2_h[t-1] = unpack(clones.lstm2[t]:backward(
                {lstm1_h[t], lstm2_c[t-1], lstm2_h[t-1]},
                {dlstm2_c[t], dlstm2_h[t]}
            ))

            dlstm1_h[t]:add(dlstm1ht)
        end

        -- backprop through LSTM timestep
        dembeddings[t], dlstm1_c[t-1], dlstm1_h[t-1] = unpack(clones.lstm1[t]:backward(
            {embeddings[t], lstm1_c[t-1], lstm1_h[t-1]},
            {dlstm1_c[t], dlstm1_h[t]}
        ))

        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate1_c:copy(lstm1_c[#lstm2_c])
    initstate1_h:copy(lstm1_h[#lstm2_h])

    initstate2_c:copy(lstm2_c[#lstm2_c])
    initstate2_h:copy(lstm2_h[#lstm2_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-2}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
    end
end


