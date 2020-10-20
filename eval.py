import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
from torchvision.transforms import Normalize, ToTensor, Compose

from components import *
from imutils import pad_img_KAR, pad_array_KAR
from vis import draw_scanpath


parser = argparse.ArgumentParser('Visual scanpath prediction')
parser.add_argument('-i', '--image', type=str, required=True, help='path to the input image')
parser.add_argument('-s', '--semantic', type=str, required=True, help='path to the semantic file')
parser.add_argument('-l', '--length', type=int, default=8, help='scanpath length')
parser.add_argument('-n', '--num_scanpaths', type=int, default=10, help='number of scanpaths to generate')
args = parser.parse_args()

NUM_SCANPATHS = args.num_scanpaths
SCANPATH_LENGTH = args.length

torch.set_grad_enabled(False)

img_orig = Image.open(args.image)
imgs, (pad_w, pad_h) = pad_img_KAR(img_orig, 400, 300)
ratio = imgs.size[0] / 400
imgs = imgs.resize((400, 300))

transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
imgs = transform(imgs).unsqueeze(0)
imgs = imgs.to(device)

sem_infos = np.load(args.semantic)
sem_infos, (_, _) = pad_array_KAR(sem_infos, 300, 400)
sem_infos = torch.LongTensor(np.int32(sem_infos)).unsqueeze(0).unsqueeze(0)
sem_infos = sem_infos.to(device)
fix_trans = torch.FloatTensor([0.19]).to(device)

y, x = np.mgrid[0:300, 0:400]
x_t = torch.from_numpy(x / 300.).float().reshape(1, 1, -1)
y_t = torch.from_numpy(y / 300.).float().reshape(1, 1, -1)
xy_t = torch.cat([x_t, y_t], dim=1).to(device)

scanpaths = list()
for scanpath_idx in range(NUM_SCANPATHS):
    first_fix = first_fix_sampler.sample()
    ob.set_last_fixation(first_fix[0], first_fix[1])
    pred_sp_x = [first_fix[0]]
    pred_sp_y = [first_fix[1]]
    pred_sp_fd = list()

    feature = feature_extractor(imgs)
    sem_infos = F.interpolate(sem_infos.float(), size=[feature.size(2), feature.size(3)]).long()
    sem_features = torch.zeros((feature.size(0), 3001, feature.size(2), feature.size(3))).float().to(device)
    sem_features[0, ...].scatter_(0, sem_infos[0, ...], 1)
    fused_feature = fuser(feature, sem_features)

    state_size = [1, 512] + list(fused_feature.size()[2:])
    ior_state = (torch.zeros(state_size).to(device), torch.zeros(state_size).to(device))
    state_size = [1, 128] + list(fused_feature.size()[2:])
    roi_state = (torch.zeros(state_size).to(device), torch.zeros(state_size).to(device))

    pred_xt = torch.tensor(np.int(pred_sp_x[-1])).float().to(device)
    pred_yt = torch.tensor(np.int(pred_sp_y[-1])).float().to(device)
    roi_map = roi_gen.generate_roi(pred_xt, pred_yt).unsqueeze(0).unsqueeze(0)
    pred_fd = fix_duration(fused_feature, roi_state[0], roi_map)
    pred_sp_fd.append(pred_fd[0, -1].item() * 750)

    for step in range(0, SCANPATH_LENGTH - 1):
        ior_state, roi_state, _, roi_latent = iorroi_lstm(fused_feature, roi_map, pred_fd, fix_trans, ior_state, roi_state)

        mdn_input = roi_latent.reshape(1, -1)
        pi, mu, sigma, rho = mdn(mdn_input)

        pred_roi_maps = MDN.mixture_probability(pi, mu, sigma, rho, xy_t).reshape((-1, 1, 300, 400))
        samples = list()
        for _ in range(30):
            samples.append(MDN.sample_mdn(pi, mu, sigma, rho).data.cpu().numpy().squeeze())

        samples = np.array(samples)
        samples[:, 0] = samples[:, 0] * 300
        samples[:, 1] = samples[:, 1] * 300
        x_mask = (samples[:, 0] > 0) & (samples[:, 0] < 400)
        y_mask = (samples[:, 1] > 0) & (samples[:, 1] < 300)
        samples = samples[x_mask & y_mask, ...]

        sample_idx = -1
        max_prob = 0
        roi_prob = pred_roi_maps.data.cpu().numpy().squeeze()
        for idx, sample in enumerate(samples):
            sample = np.int32(sample)
            p_ob = ob.prob(sample[0], sample[1])
            p_roi = roi_prob[sample[1], sample[0]]
            if p_ob * p_roi > max_prob:
                max_prob = p_ob * p_roi
                sample_idx = idx

        if sample_idx == -1:
            fix = first_fix_sampler.sample()
            pred_sp_x.append(fix[0])
            pred_sp_y.append(fix[1])
        else:
            pred_sp_x.append(samples[sample_idx][0])
            pred_sp_y.append(samples[sample_idx][1])

        ob.set_last_fixation(pred_sp_x[-1], pred_sp_y[-1])

        pred_xt = torch.tensor(np.int(pred_sp_x[-1])).float().to(device)
        pred_yt = torch.tensor(np.int(pred_sp_y[-1])).float().to(device)
        roi_map = roi_gen.generate_roi(pred_xt, pred_yt).unsqueeze(0).unsqueeze(0)
        pred_fd = fix_duration(fused_feature, roi_state[0], roi_map)
        pred_sp_fd.append(pred_fd[0, -1].item() * 750)

    pred_sp_x = [x * ratio - pad_w // 2 for x in pred_sp_x]
    pred_sp_y = [y * ratio - pad_h // 2 for y in pred_sp_y]
    scanpaths.append(np.array(list(zip(pred_sp_x, pred_sp_y, pred_sp_fd))))

    plt.imshow(img_orig)
    plt.axis('off')
    draw_scanpath(pred_sp_x, pred_sp_y, pred_sp_fd)
    plt.show()

name = os.path.basename(args.image)
name = os.path.splitext(name)[0]
np.save(f'./results/{name}.npy', scanpaths)


