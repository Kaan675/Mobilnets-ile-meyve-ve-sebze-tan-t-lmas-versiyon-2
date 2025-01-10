import os
import shutil

# Örnek veri seti klasör yapısını oluşturma
dataset_dir = 'dataset'
os.makedirs(os.path.join(dataset_dir, 'train/elma'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/muz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/brokoli'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/portakal'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/salatalık'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/domates'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/havuç'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/kereviz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/elma'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/muz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/brokoli'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/portakal'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/salatalık'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/domates'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/havuç'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/kereviz'), exist_ok=True)

# Örnek görüntüleri doğru klasörlere kopyalama
# Not: Bu adımı gerçekleştirmek için kendi görüntülerinizi kullanmalısınız.
# Örnek olarak, elma görüntülerini `train/apple` klasörüne yerleştirin.
