import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DINOLoss(nn.Module):
    """
    DINO loss from the paper "Emerging Properties in Self-Supervised Vision Transformers"
    with improvements from DINOv2
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        teacher_out = teacher_output / self.teacher_temp
        
        # Teacher sharpening
        teacher_out = F.softmax(teacher_out, dim=-1).detach()
        
        # Center the teacher output
        teacher_out = teacher_out - self.center
        
        # Student softmax
        student_out = F.log_softmax(student_out, dim=-1)
        
        # Cross entropy loss
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        # Update center for teacher output
        self.update_center(teacher_output)
        
        return loss
        
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        
        # Update center
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOHead(nn.Module):
    """
    Projection head used for DINO/DINOv2
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True):
        super().__init__()
        
        # First layer: projection
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        
        # Last FC layer
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        # Option to normalize the last layer
        self.norm_last_layer = norm_last_layer
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Normalize last layer
        if norm_last_layer:
            nn.init.constant_(self.last_layer.weight, 0)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        
        # Last layer
        if self.norm_last_layer:
            w = self.last_layer.weight.clone()
            w = F.normalize(w, dim=1, p=2)
            x = F.linear(x, w)
        else:
            x = self.last_layer(x)
            
        return x

class MultiCropWrapper(nn.Module):
    """
    Wrapper for processing multiple crops through the backbone and head
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        # Handle multiple crops
        if not isinstance(x, list):
            x = [x]
            
        outputs = []
        for crop in x:
            cls_token, _, _ = self.backbone(crop)
            outputs.append(self.head(cls_token))
            
        return outputs

class DataAugmentation:
    """
    Data augmentation for DINOv2-like training.
    Creates multiple crops of different sizes.
    """
    def __init__(self, global_crops_scale=(0.5, 1.0), local_crops_scale=(0.2, 0.5), 
                 local_crops_number=8, global_size=224, local_size=96):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_size = global_size
        self.local_size = local_size
        
        # Augmentations for global crops
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale
        ])
        
        # Augmentations for local crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale, 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale
        ])
        
    def __call__(self, image):
        """
        Apply different transformations to create multiple views of the same image
        """
        crops = []
        
        # Apply global transforms
        for _ in range(2):
            crops.append(self.global_transform(image))
            
        # Apply local transforms
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
            
        return crops

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Cosine scheduler with warmup for updating parameters like teacher momentum
    """
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    
    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train_one_epoch(student, teacher, train_loader, dino_loss, optimizer, 
                   epoch, n_epochs, warmup_teacher_temp_epochs=5, 
                   teacher_temp=0.04, student_temp=0.1, 
                   momentum_schedule=None):
    """
    One epoch of DINOv2-like training
    """
    student.train()
    teacher.eval()  # Teacher is in eval mode
    
    metric_logger = MetricLogger()
    header = f'Epoch: [{epoch}/{n_epochs}]'
    
    # Adjust teacher temperature
    teacher_temp_schedule = np.concatenate((
        np.linspace(0.07, teacher_temp, warmup_teacher_temp_epochs),
        np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp
    ))
    
    for it, (images, _) in enumerate(metric_logger.log_every(train_loader, 10, header)):
        # Update weight decay and learning rate
        it = len(train_loader) * epoch + it  # global iteration
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:  # Only the first group is regularized (typically non-bias parameters)
                param_group["weight_decay"] = 0.04
            
        # Move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        
        # Teacher and student forward passes
        teacher_output = teacher(images[:2])  # only the global views
        student_output = student(images)
        
        # Loss computation
        curr_teacher_temp = teacher_temp_schedule[epoch]
        loss = dino_loss(student_output, teacher_output, curr_teacher_temp)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update of the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
        # Log metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(teacher_temp=curr_teacher_temp)
        metric_logger.update(momentum=m)
        
    # Return the metric values averaged over the epoch
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class MetricLogger:
    """
    Utility class for logging metrics during training
    """
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)
            
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        # For multi-GPU training - not implemented in this simplified version
        pass
        
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
            
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                print(f"{i}/{len(iterable)}: {self}")

class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a window
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.values = []
        self.total = 0.0
        self.count = 0
        
    def update(self, value):
        self.values.append(value)
        self.total += value
        self.count += 1
        if len(self.values) > self.window_size:
            self.total -= self.values.pop(0)
            
    @property
    def median(self):
        return np.median(self.values)
    
    @property
    def avg(self):
        return np.mean(self.values)
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    def __str__(self):
        return f"{self.global_avg:.4f} ({self.avg:.4f})"

def main_pretraining():
    """
    Main function for DINOv2-like pretraining
    """
    import numpy as np
    import torch.distributed as dist
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    
    # Parameters
    batch_size = 64
    epochs = 100
    warmup_epochs = 10
    
    # Create model
    student = ViTEncoder(img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12)
    teacher = copy.deepcopy(student)
    student = MultiCropWrapper(student, DINOHead(in_dim=768, out_dim=65536))
    teacher = MultiCropWrapper(teacher, DINOHead(in_dim=768, out_dim=65536))
    
    # Move models to GPU
    student = student.cuda()
    teacher = teacher.cuda()
    
    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
        
    # DDP wrapper
    student = torch.nn.parallel.DistributedDataParallel(student)
    teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    
    # Data augmentation
    transform = DataAugmentation()
    
    # Dataset and dataloader
    dataset = YourMedicalDataset(transform=transform)  # Implement your dataset
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Loss function
    dino_loss = DINOLoss(out_dim=65536, teacher_temp=0.04, student_temp=0.1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=0.04, betas=(0.9, 0.999))
    
    # Scheduler
    lr_schedule = cosine_scheduler(
        base_value=5e-4,
        final_value=1e-6,
        epochs=epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=warmup_epochs,
    )
    
    # Momentum schedule
    momentum_schedule = cosine_scheduler(
        base_value=0.996,
        final_value=1.0,
        epochs=epochs,
        niter_per_ep=len(data_loader),
    )
    
    # Start training
    for epoch in range(epochs):
        # Set sampler epoch
        data_loader.sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            train_loader=data_loader,
            dino_loss=dino_loss,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=epochs,
            momentum_schedule=momentum_schedule,
        )
        
        # Save checkpoint
        if dist.get_rank() == 0:
            save_checkpoint({
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, is_best=False, filename=f'checkpoint_{epoch}.pth')
            
    # Save final model
    if dist.get_rank() == 0:
        torch.save(student.backbone.state_dict(), 'dinov2_pretrained_backbone.pth')
        
    return student.backbone

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')