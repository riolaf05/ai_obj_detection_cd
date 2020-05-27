### Info

Weight imprinting is a technique for retraining a neural network (classification models only) using a small set of sample data. It's designed to update the weights for only the last layer of the model, but in a way that can retain existing classes while adding new ones.

It has specific model architecture requirements. In the example it is used on a version of MobileNet v1 that is compatible, but if you prefer a different model, then you must make the necessary changes to your model (see [references](https://coral.ai/docs/edgetpu/retrain-classification-ondevice/#api-summary)).

To perform weight imprinting:

1. `git clone https://github.com/google-coral/edgetpu/tree/master/examples`

2. Put some photos of a new class (even just 5 - 10 photos should work) into `open_image_v4_subset/<class_name>` folder.

3. Download pretrained MobileNet v1:

```console
wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite
```

4. Run:

```console
python3 imprinting_learning.py \
--model_path mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite \
--data open_image_v4_subset \
--output retrained_imprinting_model.tflite
```

5. To try the retrained model:

```console
python3 classify_image.py \
--model retrained_imprinting_model.tflite \
--label retrained_imprinting_model.txt \
--image <new_image>.jpg
```

