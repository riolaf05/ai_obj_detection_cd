from KangarooDataset import extract_boxes, KangarooDataset
#from matplotlib import pyplot

def main():
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

    

if __name__ == "__main__":
    main()