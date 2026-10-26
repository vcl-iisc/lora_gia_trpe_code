"""
Example script to run attacks in this repository directly without simulation.
This can be useful if you want to check a model architecture and model gradients computed/defended in some shape or form
against some of the attacks implemented in this repository, without implementing your model into the simulation.

All caveats apply. Make sure not to leak any unexpected information.
"""
import torch
import torchvision
import breaching
import logging
import os
from datetime import datetime


class data_cfg_default:
    modality = "vision"
    size = (1_281_167,)
    classes = 1000
    shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(32),
        # torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
    ]
)


def main():
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"minimal_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting minimal example reconstruction...")

    setup = dict(device=torch.device("cpu"), dtype=torch.float)

    # This could be your model:
    logging.info("Loading model...")
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # And your dataset:
    logging.info("Loading dataset...")
    dataset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, transform=transforms, download=True) 
    #torchvision.datasets.ImageNet(root="~/data/imagenet", split="val", transform=transforms)
    datapoint, label = dataset[1200]  # This is the owl, just for the sake of this experiment
    labels = torch.as_tensor(label)[None, ...]

    # This is the attacker:
    logging.info("Preparing attacker...")
    cfg_attack = breaching.get_attack_config("invertinggradients")
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

    # ## Simulate an attacked FL protocol
    # Server-side computation:
    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]
    # User-side computation:
    loss = loss_fn(model(datapoint[None, ...]), labels)
    shared_data = [
        dict(
            gradients=torch.autograd.grad(loss, model.parameters()),
            buffers=None,
            metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None,),
        )
    ]

    # Attack:
    logging.info("Starting reconstruction...")
    logging.info("Starting reconstruction...")
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)
    print(reconstructed_user_data['data'].shape)

    # Save the reconstructed image
    from torchvision.transforms import ToPILImage
    reconstructed_img = reconstructed_user_data['data'].squeeze(0)  # Remove batch dimension
    
    # Denormalize the image
    mean = torch.tensor(data_cfg_default.mean).view(3, 1, 1)
    std = torch.tensor(data_cfg_default.std).view(3, 1, 1)
    reconstructed_img = reconstructed_img * std + mean
    reconstructed_img = torch.clamp(reconstructed_img, 0, 1)  # Clamp to [0, 1] range
    
    # Save the image
    os.makedirs("viz", exist_ok=True)
    img_path = os.path.join("viz", f"reconstructed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    to_pil = ToPILImage()
    pil_img = to_pil(reconstructed_img)
    pil_img.save(img_path)
    logging.info(f"Reconstructed image saved to {img_path}")
    
    logging.info("Reconstruction finished.")

    # Do some processing of your choice here. Maybe save the output image?
    # logging.info(f"Reconstruction stats: {stats}")


if __name__ == "__main__":
    main()
