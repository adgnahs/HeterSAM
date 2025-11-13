import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from models_swin.model_single import ModelEmb
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F


def norm_batch(x):
    # 归一化batch
    # 对batch进行 归一化(min - maxnormalization)，确保值在[0, 1]之间。
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    # 创建文件夹
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    # 训练步骤    criterion：损失函数   accumulation_steps：梯度累积步数，即 step+1 达到 accumulation_steps 时，才执行优化器更新      step：训练步数
    size = masks.shape[2:]  #获取mask的shape
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')  #调整gts
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)  #计算loss
    loss.backward()  #计算梯度，反向传播
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps  进行梯度累计
        optimizer.step()  #更新模型参数
        optimizer.zero_grad()  #清空梯度
    return loss.item()  #返回损失值


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam, optimizer, transform, epoch):
    # 训练单个epoch
    #ds：训练数据集   model：嵌入模型   sam   optimizer：优化器  transform：图像变换    epoch：当前epoch号
    loss_list = []#存储当前epoch的损失值
    pbar = tqdm(ds)#进度条
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    optimizer.zero_grad()#清空梯度
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        # imgs：输入  gts  original：原始大小   img sz：目标大小
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        # F.interpolate()：调整图像尺寸 以匹配 Idim（例如 512×512）
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        # model(orig_imgs_small)：提取特征嵌入
        dense_embeddings = model(orig_imgs_small)
        # get_input_dict()：将输入转换为字典格式，供 SAM 处理
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        #sam_call:将 dense_embeddings 输入到 SAM，获取分割 mask;norm_batch:归一化mask
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        #计算Dice loss和BCE loss
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        #梯度累加
        loss_list.append(loss)
        #进度条更新
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
        #返回本epoch结束时的平均loss
    return np.mean(loss_list)


def inference_ds(ds, model, sam, transform, epoch, args):
    pbar = tqdm(ds)
    model.eval()#设置为评估模式
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])

    #遍历 ds 数据集，将数据移动到 sam.device
    for imgs, gts, original_sz, img_sz in pbar:
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        #调整原始图像尺寸，并输入嵌入模型，提取特征
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)

        #转换字典，输入sam获得sam的预测mask
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))

        #调整gts和masks的大小来适配input_size和original_size
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts.unsqueeze(dim=0), input_size=input_size, original_size=original_size)

        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    model.train()
    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    #调用sam生成低分辨率mask
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)#preprocess预处理后压缩为一个batch
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def main(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ModelEmb(args=args).to(device)
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)#确保图像尺寸适配 SAM 模型的输入要求。
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD']))
    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(args, sam_trans=transform)
    elif args['task'] == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)

    #dataloader，批量大小为 args['Batch_size']，数据顺序打乱（shuffle=True）
    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)

    best = 0
    path_best = 'results/gpu' + str(args['folder']) + '/best.csv'
    f_best = open(path_best, 'w')
    for epoch in range(int(args['epoches'])):
        train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch)#调用单次epoch训练模型
        with torch.no_grad():
            IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)#验证评估模型性能
            if IoU_val > best:
                torch.save(model, args['path_best'])#如果iou超过之前最佳结果就保存当前模型的权重
                best = IoU_val
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    # 学习率
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    #批量大小
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    #训练轮数
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    #训练师使用的工作线程数
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    #验证时的工作线程数
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    #权重衰减，控制正则
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    # 设置任务名称，指明是哪个数据集或任务
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    #是否使用 深度可分离卷积（depthwise convolution）
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    #hardnet版本号
    parser.add_argument('-order', '--order', default=85, help='hardnet', required=False)
    #图像尺寸
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    #旋转角度
    parser.add_argument('-rotate', '--rotate', default=22, help='rotate', required=False)
    #缩放比例1和2
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['path'] = os.path.join('results',
                                'gpu' + folder,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    os.mkdir(args['vis_folder'])
    sam_args = {
        'sam_checkpoint': "cp/sam_vit_h.pth",
        'model_type': "vit_h",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    main(args=args, sam_args=sam_args)


