require 'nn'

local OPLU, parent = torch.class('nn.OPLU', 'nn.Module')
function OPLU:__init()
    parent.__init(self)
    self.swapped = torch.Tensor()
    self.swappedLong = torch.LongTensor() --only used in non-cuda setting
end


function OPLU:updateOutput(input)
    if input:size(input:dim()) % 2 ~= 0 then
        error("last dimension must be multiple of 2")
    end
    self.output:resizeAs(input)
    self.swapped:resize(input:nElement()/2, 2)
    
    local _input = input:view(-1, 2)
    local _output = self.output:view(-1, 2)
    self.swapped[{{},1}]:lt(_input[{{},1}], _input[{{},2}]) -- swapped = _input[1] < _input[2]
    
    self.swapped[{{},1}]:add(1) --convert [0,1] to [1,2] for 1-indexed tensors
    self.swapped[{{},2}]:fill(3):csub(self.swapped[{{},1}]) -- 3-self.swapped[1], converts [1,2] to [2,1]

    local swap = self.swapped
    if not self:_isCuda() then
        self.swappedLong:resize(self.swapped:size()):copy(self.swapped)
        swap = self.swappedLong
    end
    _output:copy(_input:gather(2, swap))

    return self.output 
end


function OPLU:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)

    local _gradOutput = gradOutput:view(-1, 2)
    local _gradInput = self.gradInput:view(-1, 2)
    
    local swap = self.swapped
    if not self:_isCuda() then
        swap = self.swappedLong
    end
    _gradInput:copy(_gradOutput:gather(2, swap))
    
    return self.gradInput
end


function OPLU:_isCuda()
    return string.sub(self:type(), 1, string.len('torch.Cuda')) == 'torch.Cuda'
end

