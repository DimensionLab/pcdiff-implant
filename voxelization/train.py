import os
import torch
import torch.optim as optim
import numpy as np
import shutil
import argparse
import time
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src import config
from src.data import collate_remove_none, collate_stack_together, worker_init_fn, SkullDataset
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, initialize_logger, AverageMeter, load_model_manual

# Optional wandb integration for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

np.set_printoptions(precision=4)

# Parse arguments BEFORE changing directory to handle paths correctly
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Set a random seed (default: 1)')
parser.add_argument('--wandb-project', type=str, default='pcdiff-implant-vox', help='wandb project name')
parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity/team name')
parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name (auto-generated if not set)')
parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging even if installed')
early_args = parser.parse_args()
# Convert config path to absolute path BEFORE changing directory
CONFIG_PATH = os.path.abspath(early_args.config)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def main():
    # Use the pre-parsed arguments
    args = early_args
    # Use the absolute config path
    cfg = load_config(CONFIG_PATH, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dev = "cuda:" + str(cfg['train']['gpu'])
    device = torch.device(dev if use_cuda else "cpu")
    input_type = cfg['data']['input_type']  # point cloud
    batch_size = cfg['train']['batch_size']  # 1
    model_selection_metric = cfg['train']['model_selection_metric']  # loss

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shutil.copyfile(CONFIG_PATH, os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    logger.info("using GPU: " + torch.cuda.get_device_name(0))

    # TensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tblogdir)

    inputs = None

    # Dataloader for training set
    train_dataset = SkullDataset(cfg['data']['train_path'], 'training', noise_stddev=cfg['data']['pointcloud_noise'])

    # Dataloader for validation set
    val_dataset = SkullDataset(cfg['data']['eval_path'], 'eval', noise_stddev=cfg['data']['pointcloud_noise'])
    vis_dataset = SkullDataset(cfg['data']['eval_path'], 'viz', noise_stddev=cfg['data']['pointcloud_noise'])

    collate_fn = collate_remove_none

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=cfg['train']['n_workers'],
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=cfg['train']['n_workers'],
                                             shuffle=True,
                                             collate_fn=collate_fn,
                                             worker_init_fn=worker_init_fn)

    vis_loader = torch.utils.data.DataLoader(vis_dataset,
                                             batch_size=1,
                                             num_workers=cfg['train']['n_workers_val'],
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             worker_init_fn=worker_init_fn)
    
    if torch.cuda.device_count() > 1:
        model = Encode2Points(cfg).to(device)
        #model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
    else:
        model = Encode2Points(cfg).to(device)

    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: %d'% n_parameter)
    
    # Initialize wandb (after model is created)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        dataset_name = cfg['data'].get('dataset', 'skull')
        wandb_name = args.wandb_name or f"{dataset_name}_ep{cfg['train']['total_epochs']}_bs{batch_size}_lr{cfg['train']['lr']}"
        
        # Create a flattened config dict for wandb
        wandb_config = {
            'dataset': dataset_name,
            'batch_size': batch_size,
            'lr': cfg['train']['lr'],
            'total_epochs': cfg['train']['total_epochs'],
            'seed': args.seed,
            'model_selection_metric': model_selection_metric,
            'grid_res': cfg['model']['grid_res'],
            'psr_sigma': cfg['model']['psr_sigma'],
            'c_dim': cfg['model']['c_dim'],
            'encoder': cfg['model']['encoder'],
            'decoder': cfg['model']['decoder'],
            'pointcloud_n': cfg['data']['pointcloud_n'],
            'pointcloud_noise': cfg['data']['pointcloud_noise'],
            'n_parameters': n_parameter,
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            config=wandb_config,
            resume='allow'
        )
        wandb.watch(model, log='all', log_freq=cfg['train']['print_every'] * 10)
        logger.info(f"Wandb logging enabled: {wandb.run.url}")
    else:
        if not WANDB_AVAILABLE:
            logger.info("Wandb not available. Install with: pip install wandb")
        else:
            logger.info("Wandb logging disabled")
    
    # load model
    try:
        # load model (suppress warning about weights_only)
        state_dict = torch.load(
            os.path.join(cfg['train']['out_dir'], 'model.pt'),
            weights_only=False
        )
        load_model_manual(state_dict['state_dict'], model)
            
        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        # load point cloud
    except:
        state_dict = dict()
    
    metric_val_best = state_dict.get('loss_val_best', np.inf)

    logger.info('Current best validation metric (%s): %.8f' % (model_selection_metric, metric_val_best))

    LR = float(cfg['train']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = state_dict.get('epoch', -1)
    it = state_dict.get('it', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()
    
    # Training statistics
    total_epochs = cfg['train']['total_epochs']
    train_start_time = time.time()
    epoch_times = []
    
    logger.info("=" * 80)
    logger.info(f"{'TRAINING CONFIGURATION':^80}")
    logger.info("=" * 80)
    logger.info(f"  Dataset: {cfg['data'].get('dataset', 'SkullData')}")
    logger.info(f"  Total epochs: {total_epochs}")
    logger.info(f"  Starting epoch: {start_epoch + 1}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {LR}")
    logger.info(f"  Samples per epoch: {len(train_dataset)}")
    logger.info(f"  Iterations per epoch: {len(train_loader)}")
    logger.info(f"  Validation every: {cfg['train']['validate_every']} epochs")
    logger.info(f"  Checkpoint every: {cfg['train']['checkpoint_every']} epochs")
    logger.info("=" * 80)
    logger.info("")
    
    # Training loop
    for epoch in range(start_epoch+1, total_epochs+1):
        epoch_start_time = time.time()
        epoch_loss = AverageMeter()
        epoch_loss_components = {}
        
        # Progress bar for batches
        pbar = tqdm(train_loader, 
                   desc=f"Epoch {epoch:03d}/{total_epochs:03d}",
                   leave=True,
                   dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            it += 1
            
            start = time.time()

            # perform one training step
            loss, loss_each = trainer.train_step(inputs, batch, model)

            # measure elapsed time
            end = time.time()
            runtime['all'].update(end - start)
            
            # Update epoch statistics
            epoch_loss.update(loss)
            if loss_each is not None:
                for k, l in loss_each.items():
                    if k not in epoch_loss_components:
                        epoch_loss_components[k] = AverageMeter()
                    epoch_loss_components[k].update(l.item())

            # Update progress bar
            pbar_info = {
                'loss': f'{loss:.4f}',
                'avg_loss': f'{epoch_loss.avg:.4f}',
            }
            pbar.set_postfix(pbar_info)

            # Log to tensorboard and wandb periodically
            if it % cfg['train']['print_every'] == 0:
                writer.add_scalar('train/loss', loss, it)
                
                # Log to wandb
                if use_wandb:
                    wandb_metrics = {
                        'train/loss': loss,
                        'train/epoch': epoch,
                        'train/iteration': it,
                        'train/lr': optimizer.param_groups[0]['lr'],
                    }
                
                if loss_each is not None:
                    for k, l in loss_each.items():
                        if l.item() != 0.:
                            writer.add_scalar('train/%s' % k, l, it)
                        
                        # Log to wandb
                        if use_wandb:
                            wandb_metrics[f'train/{k}'] = l.item()
                
                if use_wandb:
                    wandb.log(wandb_metrics, step=it)

            # Visualize some results
            if (it > 0) & (it % cfg['train']['visualize_every'] == 0):
                pbar.write(f"  → Saving visualizations at iteration {it}")
                for i, batch_vis in enumerate(vis_loader):
                    trainer.save(model, batch_vis, it, i)
                    if i >= 4:
                        break
        
        # Close progress bar
        pbar.close()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:])  # Average of last 10 epochs
        remaining_epochs = total_epochs - epoch
        eta = avg_epoch_time * remaining_epochs
        
        # Format time strings
        epoch_time_str = str(timedelta(seconds=int(epoch_time)))
        eta_str = str(timedelta(seconds=int(eta)))
        elapsed_str = str(timedelta(seconds=int(time.time() - train_start_time)))
        
        # Build summary log
        summary = f"\n{'─' * 80}\n"
        summary += f"Epoch {epoch:03d}/{total_epochs:03d} Summary:\n"
        summary += f"  Loss: {epoch_loss.avg:.6f}"
        if epoch_loss_components:
            summary += " ("
            summary += ", ".join([f"{k}={v.avg:.6f}" for k, v in epoch_loss_components.items()])
            summary += ")"
        summary += f"\n  Time: {epoch_time_str} | Elapsed: {elapsed_str} | ETA: {eta_str}\n"
        summary += f"{'─' * 80}"
        logger.info(summary)

        # Run validation
        if epoch > 0 and (epoch % cfg['train']['validate_every']) == 0:
            logger.info(f"\n{'═' * 80}")
            logger.info(f"{'VALIDATION':^80}")
            logger.info(f"{'═' * 80}")
            
            eval_dict = trainer.evaluate(val_loader, model)
            metric_val = eval_dict[model_selection_metric]
            
            # Format validation results
            val_summary = f"  Metric ({model_selection_metric}): {metric_val:.6f}\n"
            for k, v in eval_dict.items():
                if k != model_selection_metric:
                    val_summary += f"  {k}: {v:.6f}\n"
            logger.info(val_summary.rstrip())

            for k, v in eval_dict.items():
                writer.add_scalar('val/%s' % k, v, it)

            # Log validation metrics to wandb
            if use_wandb:
                wandb_val_metrics = {f'val/{k}': v for k, v in eval_dict.items()}
                wandb_val_metrics['val/epoch'] = epoch
                wandb.log(wandb_val_metrics, step=it)

            if -(metric_val - metric_val_best) >= 0:
                metric_val_best = metric_val
                improvement = "⭐ NEW BEST MODEL ⭐"
                logger.info(f"\n  {improvement}")
                logger.info(f"  Best {model_selection_metric}: {metric_val_best:.6f} (epoch {epoch})\n")
                logger.info(f"{'═' * 80}\n")
                
                state = {'epoch': epoch, 'it': it, 'loss_val_best': metric_val_best}
                state['state_dict'] = model.state_dict()
                torch.save(state, os.path.join(cfg['train']['out_dir'], 'model_best.pt'))
                
                # Log best model to wandb
                if use_wandb:
                    wandb.log({'val/best_metric': metric_val_best}, step=it)
                    # Save model as wandb artifact
                    artifact = wandb.Artifact(f'model-best-{wandb.run.id}', type='model')
                    artifact.add_file(os.path.join(cfg['train']['out_dir'], 'model_best.pt'))
                    wandb.log_artifact(artifact)
            else:
                logger.info(f"  Best {model_selection_metric}: {metric_val_best:.6f}")
                logger.info(f"{'═' * 80}\n")

        # Save checkpoint
        if (epoch > 0) & (epoch % cfg['train']['checkpoint_every'] == 0):
            state = {'epoch': epoch,
                     'it': it,
                     'loss_val_best': metric_val_best}
            pcl = None
            state['state_dict'] = model.state_dict()
                
            torch.save(state, os.path.join(cfg['train']['out_dir'], 'model.pt'))

            if (it % cfg['train']['backup_every'] == 0):
                torch.save(state, os.path.join(cfg['train']['dir_model'], '%d' % epoch + '.pt'))
                logger.info(f"  💾 Backup checkpoint saved at epoch {epoch}")
            logger.info(f"  💾 Checkpoint saved at epoch {epoch}\n")

    # Training complete
    total_time = time.time() - train_start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    
    logger.info("\n" + "=" * 80)
    logger.info(f"{'TRAINING COMPLETE':^80}")
    logger.info("=" * 80)
    logger.info(f"  Total training time: {total_time_str}")
    logger.info(f"  Total epochs: {total_epochs}")
    logger.info(f"  Best {model_selection_metric}: {metric_val_best:.6f}")
    logger.info(f"  Model saved to: {cfg['train']['out_dir']}")
    logger.info("=" * 80 + "\n")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        logger.info("Wandb run finished")


if __name__ == '__main__':
    main()
