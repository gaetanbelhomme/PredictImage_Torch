require 'torch'
require 'nn'
require 'encoder'
require 'loadData'
require 'data_reader'

-- Settings :
defaults = {
  epochs = 100,
  iters = 50,
  learning_rate = 0.20,
  batch_size = 8,
  save_path = '../NetworkSave/encoder.bin'
}

cmd = torch.CmdLine()
cmd:option('-save_path', defaults.save_path, 'Folder to save network')
cmd:option('-epochs', defaults.epochs, 'number of epochs')
cmd:option('-iters', defaults.iters, 'number of iterations per epochs')
cmd:option('-learning_rate', defaults.learning_rate, 'learning rate')
cmd:option('-batch_size', defaults.batch_size, 'batch size')

options = cmd:parse(arg)

print(options)

-- Initialization :
image_reader = data_reader()

ae = encoder()
ae:initialize()
ae:printself()

criterion = nn.MSECriterion()

trainer = nn.StochasticGradient(ae.net, criterion)

trainer.learningRate = options.learning_rate
trainer.maxIteration = options.iters

-- Training : 
for t=1, options.epochs do

  offset = (t*options.batch_size)%(train_dataset_start:size(1) - options.batch_size)
  data_set = image_reader:get_training_data(options.batch_size, offset)

  print('Epoch ' .. t)
  print('Data ', offset, '->', offset + options.batch_size)
  -- print(data_set)

  trainer:train(data_set)

end

-- Network Saving :
ae:save(options.save_path)
