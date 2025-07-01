from fastai.vision.all import *

MODEL_PATH = 'Models/EfficientNET/EfficientnetB0-hanzi_0(1).pkl'  
TEST_PATH = '/home/nm/Imagens/test'

learn = load_learner(MODEL_PATH)

test_files = get_image_files(TEST_PATH)

# Prepare files and labels, skipping unknown classes
filtered_files = []
filtered_labels = []
for f in test_files:
    label = Path(f).parent.name
    if label in learn.dls.vocab.o2i:
        filtered_files.append(f)
        filtered_labels.append(label)
    else:
        print(f"Skipping {f}: label '{label}' not in model vocab.")

if not filtered_files:
    raise ValueError("No test images match the model's vocab.")

test_dl = learn.dls.test_dl(filtered_files)
preds, _ = learn.get_preds(dl=test_dl)
pred_labels = preds.argmax(dim=1)
true_labels = tensor([learn.dls.vocab.o2i[lbl] for lbl in filtered_labels])
print("len:" + str(len(filtered_files)))  # 1 distorção + padrão

accuracy = (pred_labels == true_labels).float().mean().item()
print(f'Accuracy on test images: {accuracy*100:.2f}%')  # ---> 99%

"""
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(true_labels, pred_labels, target_names=learn.dls.vocab))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=learn.dls.vocab, yticklabels=learn.dls.vocab)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""