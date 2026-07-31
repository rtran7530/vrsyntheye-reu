#UNIR-Net
import os
import sys
import glob
import argparse

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir of input images")
    ap.add_argument("--output", required=True, help="output images dir")
    ap.add_argument("--weights", default="./Checkpoints/UNIR-Net.pt")
    ap.add_argument("--repo", default=".", help="path to the UNIR-Net repo root")
    ap.add_argument("--gamma", type=float, default=1.4, help="RCE gamma_val")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from Networks.unirNet import UNIRNet
    from Utils.rce import RCE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNIRNet().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    rce = RCE(gamma_val=args.gamma).to(device)

    paths = sorted(glob.glob(os.path.join(args.input, "**", "*.png"), recursive=True))
    print(f"processing {len(paths)} images -> {args.output}")

    for i, path in enumerate(paths, 1):
        rel = os.path.relpath(path, args.input)
        out_path = os.path.join(args.output, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(path).convert("RGB")
        x = (torch.from_numpy(np.array(img)).float() / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            y = rce(model(x))

        out = (y.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(out, "RGB").save(out_path)

        if i % 500 == 0:
            print(f"  {i}/{len(paths)}")

    print("done")


if __name__ == "__main__":
    main()