#Thanks to https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
from KangarooDataset import KangarooDataset, KangarooConfig
from mrcnn.visualize import display_instances
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
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
    print("Training the model..")
    # prepare config
    config = KangarooConfig()
    config.display()
    # define the model
    print("Defining the model..")
    model = MaskRCNN(mode='training', model_dir='./models', config=config)
    # load weights (mscoco) and exclude the output layers
    print("Loding weights..")
    model.load_weights('/home/scripts/models/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    print("Training..")
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

if __name__ == "__main__":
    main()