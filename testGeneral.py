import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='General Test Script for All Modules')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=8, help='train batch size')
parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='', help='saved model path')
parser.add_argument('--data-path', type=str, default='./dataset', help='dataset path (e.g., ./dataset)')
parser.add_argument('--dataset', type=str, default='TC')
parser.add_argument('--use-sba', action='store_true', help='Use SBA module for Original')
parser.add_argument('--use-mfm', action='store_true', help='Use MFM module for Original')
parser.add_argument('--use-adapter', action='store_true', help='Use Adapter module for Original')

# NEW ARGUMENT FOR SELECTING THE MODULE ARCHITECTURE
parser.add_argument('--module', type=str, default='Original',
                    help='Choose what module version to test: Original, FCM, GCSA, MECS, MEEM, Freq, DySample. Defaults to Original.')


def test(args, isSave=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path

    dataset = crowd.Crowd_TC(os.path.join(data_path, 'test_data'), crop_size, 1, method='val')

    if len(dataset) == 0:
        print(
            f"[Error] The test dataset is EMPTY! Please check your --data-path. Currently looking inside: {os.path.join(data_path, 'test_data')}")
        return

    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)

    print(f"[*] Loading network for module: {args.module}")

    if args.module == 'FCM':
        from network import pvt_clsFCM as TCN
        model = TCN.pvt_treeformer_fcm(pretrained=False)
    elif args.module == 'GCSA':
        from network import pvt_clsGCSA as TCN
        model = TCN.pvt_treeformer_gcsa(pretrained=False)
    elif args.module == 'MECS':
        from network import pvt_clsMECS as TCN
        model = TCN.pvt_treeformer_mecs(pretrained=False)
    elif args.module == 'MEEM':
        from network import pvt_clsMEEM as TCN
        model = TCN.pvt_treeformer_meem(pretrained=False)
    elif args.module == 'Freq':
        from network import pvt_clsFreq as TCN
        model = TCN.pvt_treeformer_freq(pretrained=False, use_sba=False, use_mfm=False)
    elif args.module == 'DySample':
        from network import pvt_clsDy as TCN
        model = TCN.pvt_treeformer_dy(pretrained=False)
    else:  # Fallback to Original
        from network import pvt_cls as TCN
        model = TCN.pvt_treeformer(pretrained=False, use_sba=args.use_sba, use_mfm=args.use_mfm,
                                   use_adapter=args.use_adapter)

    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    result = []
    R2_es = []
    R2_gt = []
    l = 0;

    # Optional: create predictions folder if it doesn't exist
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    for inputs, count, name, imgauss in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size

            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)

                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred, _ = model(crop_imgs[gs:gt])
                crop_pred = crop_pred[0]

                _, _, h1, w1 = crop_pred.size()
                crop_pred = F.interpolate(crop_pred, size=(h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16
                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask

            img_err = count[0].item() - torch.sum(outputs).item()
            R2_gt.append(count[0].item())
            R2_es.append(torch.sum(outputs).item())

            print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ",
                  torch.sum(outputs).item())
            image_errs.append(img_err)
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])

            savemat('predictions/' + name[0] + '.mat', {'estimation': np.squeeze(outputs.cpu().data.numpy()),
                                                        'image': np.squeeze(inputs.cpu().data.numpy()),
                                                        'gt': np.squeeze(imgauss.cpu().data.numpy())})
            l = l + 1

    image_errs = np.array(image_errs)

    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    R_2 = r2_score(R2_gt, R2_es)

    print('\n=============================================')
    print('Testing Module: {}'.format(args.module))
    print('{}: mae {}, mse {}, R2 {}'.format(model_path, mae, mse, R_2))
    print('=============================================\n')

    if isSave:
        save_file_name = f"test_{args.module}.txt" if args.module != 'Original' else "test.txt"
        with open(save_file_name, "w") as f:
            for i in range(len(result)):
                f.write(str(result[i]).replace('[', '').replace(']', '').replace(',', ' ') + "\n")
            f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave=True)