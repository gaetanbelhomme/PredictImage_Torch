require 'torch'
require 'nn'
require 'encoder.lua'
require 'data_reader'
npy4th = require 'npy4th'

-- Settings :
defaults = {
  batch_size = 1,
  n_batches = 1,
  save_prefix = '../Data/outputNpy'
}

cmd = torch.CmdLine()
cmd:argument('net_path', 'network to load')
cmd:option('-batch_size', defaults.batch_size, 'batch size')
cmd:option('-n_batches', defaults.n_batches, 'number of batches')
cmd:option('-save_prefix', defaults.save_prefix, 'number of batches')

options = cmd:parse(arg)
print(options)

-- Initialization :
image_reader = data_reader()

ae = encoder()
ae:initialize()
ae:load(options.net_path)

offset = (3*options.batch_size)%(train_dataset_start:size(1) - options.batch_size)
print(offset)

-- Get the output :
for i=1, options.n_batches do
  data_batch, data_out = image_reader:get_testing_data(options.batch_size, offset)
  for j=1, options.batch_size do
    local input_filename = options.save_prefix .. '/' .. 'input_' .. i .. '_' .. j .. '.npy'
    local output_filename = options.save_prefix .. '/' .. 'output_' .. i .. '_' .. j .. '.npy'

    input = data_batch[j]
    result = ae:forward(input[j])
    out = data_out[j]
    print('Input : ', input)
    print('Output : ', result)

    npy4th.savenpy(input_filename, input[j])
    npy4th.savenpy(output_filename, result[j])
    npy4th.savenpy("out.npy", out[j])
  end
end
