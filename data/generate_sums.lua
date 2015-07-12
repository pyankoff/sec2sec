require 'torch'
require 'math'

local generate_sums = {}
vocab_mapping = {['1']=1, ['2']=2, ['3']=3, ['4']=4, ['5']=5, 
        ['6']=6, ['7']=7, ['8']=8, ['9']=9, ['0']=10, ['+']=11, 
        ['=']=12, ['.']=13, ['-']=14, ['@']=15}

function generate_sums.generate()
  number_size = 3
  a = torch.random(10^number_size)
  b = torch.random(10^number_size)

  instr = "@"..tostring(a) .. '+' .. tostring(b) .. '='
  outstr = tostring(a+b) .. '.'

  orig = instr .. outstr
  x = {}
  y = {}

  for i=1,#instr do
      x[i] = vocab_mapping[instr:sub(i,i)]
      y[i] = 14
  end

  for i=1,#outstr-1 do
      x[#instr+i] = vocab_mapping[outstr:sub(i,i)]
      y[#instr+i-1] = vocab_mapping[outstr:sub(i,i)]
  end

  y[#instr+#outstr-1] = vocab_mapping[outstr:sub(#outstr,#outstr)]

  return {x, y, orig}
end

return generate_sums