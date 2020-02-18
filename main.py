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

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

for item in tqdm(json_list):
    if item['type'] == 'train':
        logger.add_scalar('train/loss_B', item['data']['loss']['B'], item['data']['iter'])
        logger.add_scalar('train/loss_M', item['data']['loss']['M'], item['data']['iter'])        
        logger.add_scalar('train/loss_C', item['data']['loss']['C'], item['data']['iter'])
        logger.add_scalar('train/loss_S', item['data']['loss']['S'], item['data']['iter'])
        logger.add_scalar('train/loss_T', item['data']['loss']['T'], item['data']['iter'])

    elif item['type'] == 'val':
        logger.add_scalar('val/box_all', item['data']['box']['all'], item['data']['iter'])
        logger.add_scalar('val/box_50',  item['data']['box']['50'] , item['data']['iter'])        
        logger.add_scalar('val/box_55',  item['data']['box']['55'] , item['data']['iter'])
        logger.add_scalar('val/box_60',  item['data']['box']['60'] , item['data']['iter'])
        logger.add_scalar('val/box_65',  item['data']['box']['65'] , item['data']['iter'])
        logger.add_scalar('val/box_70',  item['data']['box']['70'] , item['data']['iter'])
        logger.add_scalar('val/box_75',  item['data']['box']['75'] , item['data']['iter'])
        logger.add_scalar('val/box_80',  item['data']['box']['80'] , item['data']['iter'])
        logger.add_scalar('val/box_85',  item['data']['box']['85'] , item['data']['iter'])
        logger.add_scalar('val/box_90',  item['data']['box']['90'] , item['data']['iter'])
        logger.add_scalar('val/box_95',  item['data']['box']['95'] , item['data']['iter'])

        logger.add_scalar('val/mask_al', item['data']['mask']['all'], item['data']['iter'])
        logger.add_scalar('val/mask_50', item['data']['mask']['50'] , item['data']['iter'])        
        logger.add_scalar('val/mask_55', item['data']['mask']['55'] , item['data']['iter'])
        logger.add_scalar('val/mask_60', item['data']['mask']['60'] , item['data']['iter'])
        logger.add_scalar('val/mask_65', item['data']['mask']['65'] , item['data']['iter'])
        logger.add_scalar('val/mask_70', item['data']['mask']['70'] , item['data']['iter'])
        logger.add_scalar('val/mask_75', item['data']['mask']['75'] , item['data']['iter'])
        logger.add_scalar('val/mask_80', item['data']['mask']['80'] , item['data']['iter'])
        logger.add_scalar('val/mask_85', item['data']['mask']['85'] , item['data']['iter'])
        logger.add_scalar('val/mask_90', item['data']['mask']['90'] , item['data']['iter'])
        logger.add_scalar('val/mask_95', item['data']['mask']['95'] , item['data']['iter'])
