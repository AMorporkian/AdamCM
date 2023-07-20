import torch
from torch.optim.optimizer import Optimizer
from collections import deque
class AdamCM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-14,
                 weight_decay=0, amsgrad=False, buffer_size=10, decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        buffer_size=buffer_size, decay=decay)
        super(AdamCM, self).__init__(params, defaults)
        
        # Initialize momentum buffer
        self.momentum_buffer = {param: torch.zeros_like(param) for param in params}  
        
        # Initialize critical momenta buffer 
        self.cm_buffer = {param: deque(maxlen=buffer_size) for param in params}
        
    def __setstate__(self, state):
        super(AdamCM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Add current momentum to buffer
                if p in self.momentum_buffer:
                    self.momentum_buffer[p].add_(exp_avg)
                else:
                    self.momentum_buffer[p] = exp_avg.clone()
                
                # Add current momentum to critical momenta buffer
                if p in self.cm_buffer:
                    self.cm_buffer[p].append(exp_avg.clone())
                else:
                    self.cm_buffer[p] = deque(maxlen=group['buffer_size'])
                    self.cm_buffer[p].append(exp_avg.clone())
                # Decay priorities in buffer
                for m in self.cm_buffer[p]:
                    m.mul_(group['decay'])
                    
                # Aggregate critical momenta
                cm_avg = exp_avg
                for m in self.cm_buffer[p]:
                    cm_avg.add_(m) 
                if p in self.cm_buffer:
                    self.cm_buffer[p].append(exp_avg.clone())
                else:
                    self.cm_buffer[p] = deque(maxlen=group['buffer_size'])
                    self.cm_buffer[p].append(exp_avg.clone())
                cm_avg.div_(len(self.cm_buffer[p]))
                
                biased_exp_avg = cm_avg

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] / (1- beta1 ** state['step'])

                p.data.addcdiv_(-step_size, biased_exp_avg, denom)

        return loss