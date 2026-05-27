
import sys; sys.path.insert(0, "..")
sys.path.insert(0, ".")
import dataset
_, _, class_names = dataset.create_dataloaders(data_dir=sys.argv[1], batch_size=1, test_plate="P6", num_workers=1, crop_neighborhood=5)
for i, n in enumerate(class_names):
    print(f"{i}: {n}")
