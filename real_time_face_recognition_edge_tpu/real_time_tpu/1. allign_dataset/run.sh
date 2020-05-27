for N in {1..20}; do \
python3 src/align/align_dataset_mtcnn.py \
—image_size 182 \
—margin 44 \
—random_order \
—gpu_memory_fraction 0.05 \
& done