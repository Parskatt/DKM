from PIL import Image
import numpy as np
import os
import torch
from tqdm import tqdm

class HpatchesDenseBenchmark:
    """WARNING: HPATCHES grid goes from [0,n-1] instead of [0.5,n-0.5] :((((( hence all keypoints from our method must be subtracted with -0.5
    """
    def __init__(self, dataset_path) -> None:
        seqs_dir = 'hpatches-sequences-release'
        self.seqs_path = os.path.join(dataset_path,seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))

    def inside_image(self,x,w,h):
        return torch.logical_and(x[:,0] < (w-1), torch.logical_and(x[:,1] < (h-1), (x > 0).prod(dim=-1)))

    def benchmark(self, model, ransac_thr = 3.,hp240=False): # ransac_thr should perhaps be lower than 1...
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        aepes = []
        pcks = []
        for seq_idx, seq_name in tqdm(enumerate(self.seq_names), total=len(self.seq_names)):
            if seq_name[0] == 'i':
                continue
            im1_path = os.path.join(self.seqs_path,seq_name,'1.ppm')
            im1 = Image.open(im1_path)
            w1,h1 = im1.size
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path,seq_name,f'{im_idx}.ppm')
                im2 = Image.open(im2_path)
                w2, h2 = im2.size
                warp, cert = model.match(im2_path, im1_path)
                pos_a, pos_b = model.to_pixel_coordinates(warp, h2, w2, h1, w1)
                pos_a, pos_b = pos_a.double().reshape(-1,2) - 0.5, pos_b.double().reshape(-1,2) - 0.5
                inv_homography = torch.from_numpy(np.loadtxt(os.path.join(self.seqs_path,seq_name, "H_1_" + str(im_idx)))).to(device)
                homography = torch.linalg.inv(inv_homography)
                pos_a_h = torch.cat([pos_a, torch.ones([pos_a.shape[0], 1],device=device)], dim=1)
                pos_b_proj_h = (homography@pos_a_h.t()).t()
                pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]
                mask = self.inside_image(pos_b_proj,w1,h1)
                residual = pos_b - pos_b_proj
                if hp240:
                    residual = torch.stack((residual[:,0]*(240/w2),residual[:,1]*(240/h2)),dim=1)
                dist = (residual ** 2).sum(dim=1).sqrt()[mask]
                aepes.append(torch.mean(dist).item())
                pck1 = (dist<1.).float().mean().item()
                pck3 = (dist<3.).float().mean().item()
                pck5 = (dist<5.).float().mean().item()
                pcks.append([pck1,pck3,pck5])
        m_pcks = np.mean(np.array(pcks),axis=0)
        m_aepe = np.mean(aepes)
        aepe_1 = np.mean(aepes[::5])
        aepe_2 = np.mean(aepes[1::5])
        aepe_3 = np.mean(aepes[2::5])
        aepe_4 = np.mean(aepes[3::5])
        aepe_5 = np.mean(aepes[4::5])
        
        return {'hp_aepe':m_aepe,'hp_pck1':m_pcks[0],'hp_pck3':m_pcks[1],'hp_pck5':m_pcks[2]}