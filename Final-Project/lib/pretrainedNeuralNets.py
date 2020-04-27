import torchvision
import torchvision.transforms as T

def getModel(num_classes, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                               num_classes=num_classes)

    return model

def getCNNFeatureExtractVGG19():
    return torchvision.models.vgg19(pretrained=True).features

if __name__ == '__main__':
    model = torchvision.models.vgg19(pretrained=True)
    torchvision.models.vgg19()


