
import torch 

from .DRFuserNo_Attention import DRFuserNo_Attention
from .DRFuser_additive import DRFuser_additive
from .DRFuser_self_attention import DRFuser_self_attention


class DRFuser():
    
        
    def build(args):
                
        if args.model_id == 'self-Attention':
            model = DRFuser_self_attention(args)
        elif args.model_id == 'No-Attention':
            model = DRFuserNo_Attention(args)
        elif args.model_id == 'Additive':
            model = DRFuser_additive(args)
        else:
            raise NotImplementedError

        model.cuda(args.device)    
        print("Using DRFuser with",args.model_id," and backbone ResNet:",args.resnet)
        return model
        