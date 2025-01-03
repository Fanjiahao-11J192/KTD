from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samus.build_sam_us import samus_model_registry
import torchvision.models as models
import torch
import timm
def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

def get_classifier(opt=None):
    if opt.classifier_name =="Resnet18":
        assert opt.classifier_size == 256, "图像尺寸需要为256"
        classifier = models.resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, opt.classifier_classes)
    elif opt.classifier_name =="Vit":
        # classifier = models.VisionTransformer(image_size=args.img_size,patch_size=16,num_layers=4,num_heads=4,hidden_dim=768,mlp_dim=3072,num_classes=args.classifier_classes)
        assert opt.classifier_size == 224, "图像尺寸需要为224"
        classifier = timm.models.vit_base_patch16_224(pretrained=True)
        classifier.head = torch.nn.Linear(classifier.head.in_features, opt.classifier_classes)

    else:
        raise RuntimeError("Could not find the classifier:", opt.classifier_name)
    return classifier