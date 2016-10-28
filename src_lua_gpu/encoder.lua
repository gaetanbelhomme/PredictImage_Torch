require 'torch'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'cunn'

encoder = {}
encoder.__index = encoder

setmetatable(encoder, {
	 __call = function (cls, ...)
	return cls.new(...)
	end,
})

function encoder.new()
	local self = setmetatable({}, encoder)
	return self
end

function encoder:initialize()
	-- input/output dimension
	nfeats = 1
	
	-- hidden units, filter sizes 
	nstates = { 32, 64, 2048}
	-- nstates = { 1, 12, 1568}
	filtsize = 5
	poolsize = 2
	pad = 2

	-- pooling :
	local pool_layer1 = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
	local pool_layer2 = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)

	-- convolution neural network (conv + relu + pool)*2 + fc + fc + unpooling
    self.net = nn.Sequential()

    -- Stage 1 : CONV & MAXPOOL : 1*64*64 -> 32*32*32
    self.net:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize, 1, 1, pad, pad))
    self.net:add(nn.ReLU())
    self.net:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize, filtsize, 1, 1, pad, pad))
  	self.net:add(nn.ReLU())

    self.net:add(pool_layer1)
        
    -- Stage 2 : CONV & MAXPOOL : 32*32*32 -> 64*16*16
    self.net:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize, 1, 1, pad, pad))
    self.net:add(nn.ReLU())
    self.net:add(pool_layer2)
        
    -- Stage 3 : FC : 64*16*16 -> 1*2048
    self.net:add(nn.Reshape(nstates[2]*16*16))
    self.net:add(nn.Dropout(0.5))
    self.net:add(nn.Linear(nstates[2]*16*16, nstates[3]))
        
	-- Stage 4 : FC : 1*2048 -> 64*16*16
	self.net:add(nn.Linear(nstates[3], nstates[2]*16*16))
	self.net:add(nn.Reshape(nstates[2],16, 16))

	-- Stage 5 : CONV & MAXUNPOOL : 64*16*16 -> 32*32*32
	self.net:add(nn.SpatialMaxUnpooling(pool_layer2))
	self.net:add(nn.SpatialConvolutionMM(nstates[2], nstates[1], filtsize, filtsize, 1, 1, pad, pad))
	self.net:add(nn.ReLU())
	

	-- Stage 6 : CONV & MAXUNPOOL : 32*32*32 -> 1*64*64
	self.net:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize, filtsize, 1, 1, pad, pad))
  	self.net:add(nn.ReLU())
  	self.net:add(nn.SpatialMaxUnpooling(pool_layer1)) 
	self.net:add(nn.SpatialConvolutionMM(nstates[1], nfeats, filtsize, filtsize, 1, 1, pad, pad))

	self.net = self.net:cuda()
	

end

function encoder:printself()
  	print(self.net)
end

function encoder:save(filename)
  	torch.save(filename, self.net)
end

function encoder:load(filename)
  	self:initialize()
  	self.net = torch.load(filename)
end

function encoder:forward(input)
  	return self.net:forward(input)
end
