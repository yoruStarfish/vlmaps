# 檔案位置: /home/robotic/vlmaps/run_build_map.py
import sys
import os
import argparse



# --- 2. 放上我們的終極掉包魔法！ ---
import timm
backup_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth'
if 'vit_large_patch16_384' in timm.models.vision_transformer.default_cfgs:
    timm.models.vision_transformer.default_cfgs['vit_large_patch16_384']['url'] = backup_url

# --- 3. 載入必要的套件 ---
import math
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import clip

from utils.mapping_utils import load_pose, save_map, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point
from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image

def load_depth(depth_filepath):
    # 這裡改成支援我們 data_collector 存的 cv2 png 格式
    depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
    # RealSense 存的 16-bit 深度圖單位通常是毫米(mm)，這裡轉成公尺(m)給 VLMaps 算
    depth = depth.astype(np.float32) / 1000.0
    return depth

# --- 4. 實機專用版 (去除了 GT Semantic) ---
def create_lseg_map_batch(img_save_dir, camera_height, cs=0.05, gs=1000, depth_sample_rate=100): # depth_sample_rate 是**「深度圖的降採樣率」，說白了就是「每幾個點才挑一個點來算」**
    # 假設隨便拍一張照片，裡面就有超過 30 萬個像素（點）。如果你用 Python 的 for 迴圈把這 30 萬個點全部轉成 3D 座標再投影到地圖上，處理一張照片可能就要好幾分鐘，這對實機來說是完全無法接受的！
    # 所以我們引入了 depth_sample_rate 這個參數，讓你可以控制「每幾個點才挑一個點來算」，大幅降低計算量，讓地圖建立的速度提升到秒級別！當然，這也會犧牲一些細節，但在實機上通常是值得的。
    mask_version = 1 
    crop_size = 480 
    base_size = 520 
    lang = "door,chair,table,ground,ceiling,other"
    labels = lang.split(",")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version) 
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    
    model = LSegEncNet(lang, arch_option=0, block_depth=0, activation='lrelu', crop_size=crop_size)
    
    # 解決路徑問題：改成絕對路徑
    model_state_dict = model.state_dict()
    checkpoint_path = "/home/robotic/vlmaps/lseg/checkpoints/demo_e200.ckpt"
    pretrained_state_dict = torch.load(checkpoint_path)
    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict, strict=False)
    model.eval()
    model = model.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print(f"Loading real-world scene data from: {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    pose_dir = os.path.join(img_save_dir, "pose")

    # 確保資料夾存在
    if not os.path.exists(rgb_dir):
        print(f"❌ 找不到 RGB 資料夾: {rgb_dir}")
        return

    rgb_list = sorted([os.path.join(rgb_dir, x) for x in os.listdir(rgb_dir)])
    depth_list = sorted([os.path.join(depth_dir, x) for x in os.listdir(depth_dir)])
    pose_list = sorted([os.path.join(pose_dir, x) for x in os.listdir(pose_dir)])

    map_save_dir = os.path.join(img_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy") # 用來畫出彩色地圖用來畫出彩色俯視圖（給人看的）
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy") # 用來記錄每個格子裡的 LSeg AI 特徵（512 維度的浮點數矩陣），這是給 VLMaps 算路徑規劃用的
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy") # 用來記錄目前畫布上每個格子裡有多少點的 CLIP 特徵被融合進去（避免被地板的顏色蓋過桌子的顏色）
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32) #  用來記錄目前畫布上每個格子的「最高高度」（避免被地板的顏色蓋過桌子的顏色）
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32) # 來儲存每個格子的 LSeg AI 特徵(512 維度的浮點數矩陣）
    obstacles = np.ones((gs, gs), dtype=np.uint8) # 用來記錄障礙物的二值化地圖（1 代表有障礙物，0 代表沒有障礙物）。初始值設為 1，表示一開始我們假設整個地圖都是有障礙物的，然後隨著點雲資料的加入，我們會把那些確定沒有障礙物的格子標記為 0。
    weight = np.zeros((gs, gs), dtype=float) # 用來記錄這個格子被「蓋了幾次印章」，之後用來算特徵的平均值

    tf_list = []
    # 移除了 semantic 的 zip
    data_iter = zip(rgb_list, depth_list, pose_list)
    pbar = tqdm(total=len(rgb_list))
    
    for rgb_path, depth_path, pose_path in data_iter:
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 這裡的 load_pose 我們假設存檔格式是單純的 4x4 txt 矩陣
        # 如果格式有問題我們後續可以微調這裡
        pose = np.loadtxt(pose_path)
        tf_list.append(pose)
        if len(tf_list) == 1: # 這裡的邏輯是：第一幀的 pose 就當作全局座標系的原點，後續的 pose 都要相對於第一幀來轉換 （也就是說，第一幀的 pose 會被轉成單位矩陣，第二幀的 pose 會被轉成「從第一幀到第二幀的相對變換」，第三幀的 pose 會被轉成「從第一幀到第三幀的相對變換」，以此類推）
            # 這是因為ratbmap一開始打開的時候，座標可能不是完美的 (0,0,0)，而是某個隨機的 pose。為了讓地圖建立的過程更穩定，我們把第一幀的 pose 當作全局座標系的原點，這樣後續的 pose 就會相對於第一幀來轉換，確保整個地圖建立的過程中，座標系是一致的。
            init_tf_inv = np.linalg.inv(tf_list[0]) 

        tf = init_tf_inv @ pose # 把當前幀的 pose 轉換到以第一幀為原點的全局座標系下 (camera frame to world frame)
        depth = load_depth(depth_path)

        pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size) # 透過 LSeg 模型，算出這張 RGB 圖的每個像素對應的語義特徵（512 維度的浮點數矩陣）。這裡得到的 pix_feats 是一個形狀為 (1, 512, H', W') 的張量，其中 H' 和 W' 是經過 LSeg 模型處理後的特徵圖尺寸，通常會比原始 RGB 圖小一些（例如原始 RGB 圖是 480x640，LSeg 的特徵圖可能是 120x160）。我們後續會把這些特徵投影到地圖上，讓 VLMaps 可以利用這些語義資訊來做更聰明的路徑規劃。
        
        pc, mask = depth2pc(depth) # 透過深度圖，算出相機視角下的 3D 點雲
        shuffle_mask = np.arange(pc.shape[1]) 
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate] # 依照這個比例抽取。如果 depth_sample_rate = 100，
        # 代表它每 100 個點只隨機挑 1 個點來用（30 萬點瞬間變成 3 千點）
        # 因為 VLMaps 的地圖解析度（例如 5cm x 5cm 的網格）並不需要那麼密集的點雲，這樣做可以在幾乎不影響建圖品質的前提下，把運算速度提升 100 倍
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf) # 把相機視角下的點雲，轉換成整個地圖的世界座標系下的點雲。這樣我們就知道每個點在整個地圖上的位置了，接下來就可以把這些點投影到地圖的格子上，並且把它們對應的 RGB 顏色和 LSeg 語義特徵也投影到同一個格子裡，讓 VLMaps 可以利用這些資訊來做路徑規劃和導航了。    

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)): # 把每一個 3D 點 (p_local 和 p)，投影回 2D 的地圖網格 (x, y) 上
            x, y = pos2grid_id(gs, cs, p[0], p[2])

            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5: # 這裡的邏輯是：如果這個點投影到地圖上的格子索引超出地圖的邊界了，或者這個點的高度（y）太低了（例如低於 -0.5 米，可能是測量誤差或是地板以下的點），那我們就直接跳過這個點，不把它投影到地圖上了。這樣做可以避免一些不合理的點對地圖造成干擾，讓地圖看起來更乾淨、更準確。
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            # 防呆：確保像素索引沒有超出邊界
            if rgb_py >= rgb.shape[0] or rgb_px >= rgb.shape[1]: # 這裡的邏輯是：如果這個點投影回 RGB 圖上的像素索引超出 RGB 圖的邊界了，那我們就直接跳過這個點，不去讀取它的顏色了。這樣做可以避免程式崩潰，因為如果我們試圖讀取一個不存在的像素，就會發生索引錯誤。
                continue
                
            rgb_v = rgb[rgb_py, rgb_px, :]
            
            # in camera frame，y is the height and y軸朝下 （所以y越小代表越高）
            if p_local[1] < color_top_down_height[y, x]: # 如果現在這個點比之前畫在同一個格子上的點還要「高」（也就是說，這個點更有可能是桌子、椅子等物體，而不是地板），那我們就用這個點的顏色來更新這個格子的顏色，讓地圖看起來更像真實世界的樣子，而不是一片綠色的地板（因為地板通常是最低的，所以如果我們不做這個判斷，很多格子就會被地板的顏色蓋掉，看起來就很難看了）
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]

            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1) # 這裡的邏輯是：如果這個格子之前已經有特徵了，那就把新的特徵和舊的特徵做平均，讓它們融合在一起；如果這個格子之前沒有特徵，那就直接把新的特徵放進去。這樣做可以讓地圖上的每個格子都能夠融合來自不同幀的資訊，讓地圖更完整、更穩定。
                weight[y, x] += 1
            
            if p_local[1] > camera_height: # 如果這個點的高度（y）比相機的高度還要高，那我們就認為這個格子裡有障礙物了，因為相機不可能看到比自己還高的點（除非是測量誤差），所以這些點很可能是牆壁、桌子等障礙物。這樣做可以讓地圖上標記出那些有障礙物的格子，讓 VLMaps 在做路徑規劃的時候能夠避開這些格子，找到一條安全的路徑。
                continue
            obstacles[y, x] = 0 # 高度低於障礙物，代表這個格子裡沒有障礙物了，把它標記為 0
            
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)
    print("✅ 地圖建立完成並儲存成功！")


def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    # ...(這裡維持你原本的實作不變，直接複製貼上即可)...
    vis_image = image.copy()
    image = transform(image).unsqueeze(0).cuda()
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5
    
    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
            count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        logits_outputs = logits_outputs[:,:,:height,:width]
    outputs = outputs.cpu()
    outputs = outputs.numpy() 
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"開始處理來自 {args.data_dir} 的資料...")
    create_lseg_map_batch(args.data_dir, camera_height=0.69)