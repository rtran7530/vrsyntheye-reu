# UnReflectAnything 
import argparse
import unreflectanything as u


def apply_unreflect(input_dir, output_dir, weights_path,
                    batch_size=4, threshold=0.3, dilation=15,
                    composite=True, device="cuda"):
    model = u.model(pretrained=True, weights_path=weights_path, device=device)
    u.inference(input_dir, output=output_dir, model=model,
                batch_size=batch_size, threshold=threshold,
                dilation=dilation, composite=composite, show_progress=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_dir", required=True)
    p.add_argument("--out", dest="output_dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--dilation", type=int, default=15)
    p.add_argument("--composite", action="store_true", default=True)
    a = p.parse_args()
    apply_unreflect(a.input_dir, a.output_dir, a.weights,
                    a.batch_size, a.threshold, a.dilation, a.composite)