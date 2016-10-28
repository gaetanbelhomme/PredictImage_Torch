require 'LoadData'

data_reader = {}
data_reader.__index = data_reader

setmetatable(data_reader, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})


function data_reader.new()
  local self = setmetatable({}, data_reader)
  self.dims = {1,64,64}
  return self
end


function data_reader:get_training_data(num_elements, offset)
  dataset = {}
  function dataset:size() return num_elements end

  for i=1,num_elements do
      dataset[i] = {train_dataset_start[offset + i], train_dataset_end[offset + i]}
      dataset[i][1] = dataset[i][1]:double()
      dataset[i][2] = dataset[i][2]:double()
  end

  return dataset
end


function data_reader:get_testing_data(num_elements, offset)
  dataset = {}
  function dataset:size() return num_elements end

  for i=1,num_elements do
      dataset[i] = {test_dataset_start[offset + i] }
      dataset[i][1] = dataset[i][1]:double()
  end

  return dataset
end