import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from src.trainer import GeneratorModule
from src.utils import get_config


def torch_to_image_numpy(tensor: torch.Tensor):
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0) for i in range(tensor.shape[0])]
    return im_np


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # Optional: To print the full config for debugging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs_per_sample = cfg.generator.runs_per_sample

    path_checkpoint = Path(cfg.trainer.checkpoint + cfg.generator.ckpt)

    output_path = Path(
        cfg.generator.output
        + f"\\{cfg.project_name}"
        + f"\\power{cfg['dataset']['power']}_{cfg['model']['loss']}_{cfg['generator']['generator_mode']}"
    )
    print(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    module = GeneratorModule.load_from_checkpoint(
        checkpoint_path=path_checkpoint,
        strict=False,
        config=cfg,
        use_fp16=cfg.trainer.fp16,
        timestep_respacing=str(cfg.generator.timestep_respacing),
    )
    module = module.to(device)
    module.eval()

    diffusion = module.diffusion
    model = module.model
    model = model.to(device)

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg.generator.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    shape = (
        cfg.trainer.batch_size * 2,
        module.n_channels,
        module.size_image,
        module.size_image,
    )

    generator_model = cfg["generator"]["generator_mode"]

    # generate images
    dls = [module.test_dataloader()]
    names = ["test"]

    if cfg.generator.include_val:
        dls.append(module.val_dataloader())
        names.append("val")
    if cfg.generator.include_train:
        dls.append(module.train_dataloader())
        names.append("train")

    try:
        number_of_examples = (len(dls[0]) + len(dls[1])) * cfg.trainer.batch_size
    except:
        number_of_examples = len(dls[0]) * cfg.trainer.batch_size

    print(f"Total number of examples: {number_of_examples}")

    for i, dl in enumerate(dls):
        for batch_idx, batch in tqdm(enumerate(dl), colour="blue"):

            if batch_idx == cfg["generator"]["max_batch"]:
                break
            images = []
            generated_images = []
            dirty_noisy_list = []
            sky_indexes_list = []

            im_in = batch["true"]
            im_in_ = im_in.to(device)

            dirty_noisy = batch["dirty_noisy"].to(device)
            filenames = batch["filename"]

            for _ in tqdm(range(runs_per_sample)):
                im_in = im_in_

                with torch.no_grad():
                    zero_label_noise = torch.zeros_like(dirty_noisy, device=device)
                    dirty_noisy = torch.cat([dirty_noisy, zero_label_noise], dim=0)
                    if generator_model == "ddpm":
                        im_out = diffusion.p_sample_loop(
                            model_fn,
                            cond=dirty_noisy,
                            shape=shape,
                            device=device,
                            clip_denoised=True,
                            progress=False,
                            cond_fn=None,
                        )[: cfg.trainer.batch_size]
                        dirty_noisy = dirty_noisy[: cfg.trainer.batch_size]
                    else:
                        pass

                im_in = torch_to_image_numpy(im_in)
                im_out = torch_to_image_numpy(im_out)
                dirty_noisy_ = torch_to_image_numpy(dirty_noisy)

                images.extend(im_in)
                generated_images.extend(im_out)
                dirty_noisy_list.extend(dirty_noisy_)
                sky_indexes_list.extend(filenames)

            images = np.array(images)
            generated_images = np.array(generated_images)
            dirty_noisy_list = np.array(dirty_noisy_list)

            np.save(output_path / f"batch={batch_idx}_{names[i]}_images.npy", images)
            np.save(
                output_path / f"batch={batch_idx}_{names[i]}_generated_images.npy",
                generated_images,
            )
            np.save(
                output_path / f"batch={batch_idx}_{names[i]}_dirty_noisy.npy",
                dirty_noisy_list,
            )
            np.save(
                output_path / f"batch={batch_idx}_{names[i]}_sky_indexes.npy",
                sky_indexes_list,
            )


if __name__ == "__main__":
    main()
