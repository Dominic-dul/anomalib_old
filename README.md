This is a repository for integrating Asymmetric Student Teacher network into EfficientAD network for anomaly detection in industrial manufacturing.
The code is configured to run on both CPU and CUDA environments

Launching instructions:
1. Download the MvTec dataset and extract it:
   - If it is MvTec AD dataset, extract it to a folder named 'mvtec_anomaly_detection'
     ```
     mkdir mvtec_anomaly_detection
     cd mvtec_anomaly_detection
     wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
     tar -xvf mvtec_anomaly_detection.tar.xz
     cd ..
     ```
   - If it is MvTec AD dataset, extract it to a folder name 'mvtec_loco_anomaly_detection'
     ```
     mkdir mvtec_loco_anomaly_detection
     cd mvtec_loco_anomaly_detection
     wget https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz
     tar -xf mvtec_loco_anomaly_detection.tar.xz
     cd ..
     ```
2. After the dataset is extracted, run the preprocess.py script to extract the features for teacher training.
   ```
   python3 preprocess.py --dataset_dir mvtec_anomaly_detection
   ```
      - --dataset_dir: directory of MvTec dataset images (default: mvtec_anomaly_detection)
   - After the script is run, folder './data/features' should be created with 'train' and 'test' .npy files for all categories
3. After features are extracted, run the 'train_nf_teacher.py' file to train the teacher model in the following way:
   ```
   python3 train_nf_teacher.py --subdataset bottle --train_steps 240 --dataset_dir mvtec_anomaly_detection
   ```
     - --subdataset: category of MvTec dataset (default: bottle)
     - --train_steps: number of epochs for training (default: 240)
     - --dataset_dir: directory of MvTec dataset (default: mvtec_anomaly_detection)
4. After the training of the teacher is complete, the teacher model for the specified category should be created under folder ./models with the name 'teacher_nf_{subdataset}.pth'
5. To train the student-autoencoder pair, a folder of at least 200 images from the ImageNet dataset is necessary.
   - You can download a .zip file containing a set of 1000 images from ImageNet dataset with the following link:
   ```
   https://drive.google.com/uc?export=download&id=11zBF5L5oYNu7qO1ZPCMAoLaNVGu6Sqvr
   ```
   - Extract the .zip file and store it in the root directory (suggested folder structure: imagenet_pictures/collected_images/{images...})
6. To train the student-autoencoder, run the 'efficientad.py' script in the following way:
   ```
   python3 efficientad.py --subdataset screw --test_only True --train_steps 100 --dataset_dir mvtec --imagenet_dir imagenet_pictures/collected_images
   ```
     - --subdataset: category of MvTec dataset (default: bottle)
     - --test_only: flag to whether only perform the evaluation (teacher, student, and autoencoder must be under ./models folder in advance for the 'True' flag to work) (default: False)
     - --train_steps: number of epochs for training (default: 100)
     - --dataset_dir: directory of MvTec dataset (default: mvtec_anomaly_detection)
     - --imagenet_dir: directory of images from ImageNet (default: imagenet_picures/collected_images)
7. After training and evaluation of the student-autoencoder pair and teacher are complete, the results should be saved in the corresponding folders:
  - ./models: folder, containing .pth models
  - ./output/results/{subdataset}/teacher_metrics.txt: teacher AUC mean, AUC max, loss difference between first and last epochs, training time
  - ./output/results/{subdataset}/efficientad_metrics.txt: AUC, F1, Recall, Precision scores on image and pixel level, average processing latency
  - ./output/results/{subdataset}/graphs/: loss curves of teacher and efficientad model, ROC, and Precision-Recall curves on image and pixel levels
  - ./output/results/{subdataset}/images/heat_maps: generated anomaly heatmaps for images
  - ./output/results/{subdataset}/images/masks: generated predicted masks for images
