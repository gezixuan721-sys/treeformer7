import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from network import pvt_cls as TCN
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=8, help='train batch size')
parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the train image')
parser.add_argument('--model-path', type=str,
                    default='/media/hznu-303/sdc/gzt/my_project/TreeFormer/TreeFormer-main/best_model.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str, default='/media/hznu-303/sdc/gzt/my_project/TreeFormer/Data',
                    help='dataset path')
parser.add_argument('--dataset', type=str, default='TC')


def test(args, isSave=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path

    dataset = crowd.Crowd_TC(os.path.join(data_path, 'test_data'), crop_size, 1, method='test')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)

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
    for inputs, count, name, imgauss in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            b, c, h, w = inputs.size()

            # Since the image is already 256x256, directly predict
            crop_pred, _ = model(inputs)
            crop_pred = crop_pred[0]

            # Interpolate to original size
            _, _, h1, w1 = crop_pred.size()
            outputs = F.interpolate(crop_pred, size=(h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16

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

    print('{}: mae {}, mse {}, R2 {}\n'.format(model_path, mae, mse, R_2))

    if isSave:
        with open("test.txt", "w") as f:
            for i in range(len(result)):
                f.write(str(result[i]).replace('[', '').replace(']', '').replace(',', ' ') + "\n")
            f.close()


if __name__ == '__main__':
    parser.add_argument('--use-sba', action='store_true', help='use sba module')
    parser.add_argument('--use-mfm', action='store_true', help='use mfm module')
    parser.add_argument('--use-adapter', action='store_true', help='use adapter module')
    args = parser.parse_args()
    test(args, isSave=True)

