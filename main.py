from tensorboardX import SummaryWriter
import json
import sys
import os
from tqdm import tqdm

yolact_log_path = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir)

with open(yolact_log_path, "r") as f:
    lines = f.readlines()

json_list = [json.loads(line) for line in lines]

logger = SummaryWriter(out_dir)

for item in tqdm(json_list):
    if item['type'] == 'train':
        iteration = item['data']['iter']
        logger.add_scalar('train/loss_B', item['data']['loss']['B'], iteration)
        logger.add_scalar('train/loss_M', item['data']['loss']['M'], iteration)        
        logger.add_scalar('train/loss_C', item['data']['loss']['C'], iteration)
        logger.add_scalar('train/loss_S', item['data']['loss']['S'], iteration)
        logger.add_scalar('train/loss_T', item['data']['loss']['T'], iteration)

    elif item['type'] == 'val':
        iteration = item['data']['iter']
        logger.add_scalar('val_box/all', item['data']['box']['all'], iteration)
        logger.add_scalar('val_box/50',  item['data']['box']['50'] , iteration)        
        logger.add_scalar('val_box/55',  item['data']['box']['55'] , iteration)
        logger.add_scalar('val_box/60',  item['data']['box']['60'] , iteration)
        logger.add_scalar('val_box/65',  item['data']['box']['65'] , iteration)
        logger.add_scalar('val_box/70',  item['data']['box']['70'] , iteration)
        logger.add_scalar('val_box/75',  item['data']['box']['75'] , iteration)
        logger.add_scalar('val_box/80',  item['data']['box']['80'] , iteration)
        logger.add_scalar('val_box/85',  item['data']['box']['85'] , iteration)
        logger.add_scalar('val_box/90',  item['data']['box']['90'] , iteration)
        logger.add_scalar('val_box/95',  item['data']['box']['95'] , iteration)

        logger.add_scalar('val_mask/all', item['data']['mask']['all'], iteration)
        logger.add_scalar('val_mask/50',  item['data']['mask']['50'] , iteration)        
        logger.add_scalar('val_mask/55',  item['data']['mask']['55'] , iteration)
        logger.add_scalar('val_mask/60',  item['data']['mask']['60'] , iteration)
        logger.add_scalar('val_mask/65',  item['data']['mask']['65'] , iteration)
        logger.add_scalar('val_mask/70',  item['data']['mask']['70'] , iteration)
        logger.add_scalar('val_mask/75',  item['data']['mask']['75'] , iteration)
        logger.add_scalar('val_mask/80',  item['data']['mask']['80'] , iteration)
        logger.add_scalar('val_mask/85',  item['data']['mask']['85'] , iteration)
        logger.add_scalar('val_mask/90',  item['data']['mask']['90'] , iteration)
        logger.add_scalar('val_mask/95',  item['data']['mask']['95'] , iteration)
