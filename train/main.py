from KangarooDataset import extract_boxes, KangarooDataset, KangarooConfig
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
#from matplotlib import pyplot

def main():
    # 1) LOADING TRAIN AND TEST DATASET
    # train set
    train_set = KangarooDataset()
    train_set.load_dataset('kangaroo', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # test/val set
    test_set = KangarooDataset()
    test_set.load_dataset('kangaroo', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # plot image
    #pyplot.imshow(image)
    # plot mask
    #pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
    #pyplot.show()

    '''
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        image = train_set.load_image(i)
        pyplot.imshow(image)
        # plot all masks
        mask, _ = train_set.load_mask(i)
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
    # show the figure
    pyplot.show()
    '''

    # 2) TRAINING THE MODEL 
    # prepare config
    config = KangarooConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('/home/scripts/models/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

if __name__ == "__main__":
    main()