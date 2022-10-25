import torch.nn as nn
import torch.nn.functional as F

class Conv2dSamePadding(nn.Conv1d):
    def __init__(self,
                 *args,
                 **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        padding = (-self.stride[0] + self.dilation[0]*(self.kernel_size[0]-1) + 1 +
                   inputs.size(-1) * (self.stride[0] - 1)) \
                  // 2
        return self._conv_forward(F.padd(inputs, (padding, padding)),
                                  self.weight,
                                  self.bias,
                                  )

class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
