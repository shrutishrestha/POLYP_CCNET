def transform(self,image,label):
            ia.seed(1)
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            augmentation_for_both = iaa.Sequential(
            [
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.5), # vertically flip 20% of all images
                iaa.Rotate(-45, 45),
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(scale=(0.5, 2.0))
                )])
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    aa.GammaContrast((0.2, 1.8))
                    ])
                isss.Sometimes(
                    0.3,
                    iaa.AverageBlur(k=9)
                )

            segmap = SegmentationMapsOnImage(label, shape=image.shape)
            image, label = augmentation_for_both(image = image, segmentation_maps=segmap)
            label = label.get_arr()

            return image,label

#true2
def transform(self,image,label):
    ia.seed(1)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augmentation_for_both = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # iaa.Rotate(-45, 45),
        iaa.Sometimes(
            0.5,
            iaa.Affine(scale=(0.5, 2.0))
        )])
    augmentation_for_image = iaa.Sequential(
        [
        iaa.OneOf([
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.GammaContrast((0.2, 1.8))
            ]),
        iaa.Sometimes(
            0.3,
            iaa.AverageBlur(k=9)
        )])
    image = augmentation_for_image(image = image)

    segmap = SegmentationMapsOnImage(label, shape=image.shape)
    image, label = augmentation_for_both(image = image, segmentation_maps=segmap)
    label = label.get_arr()

    return image,label