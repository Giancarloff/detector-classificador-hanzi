from fastai.vision.all import *

"""
Modelo com o set completo, único VIT que consegui salvar o pkl,
não consegui usar o pth, resultados sem valor dado o treino escasso"""

MODEL_PATH = 'Models/VIT/VIT_0.pkl'
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

print("\n" + str(len(filtered_files)) + "\n")  # ---> 8104 (FULL)

if not filtered_files:
    raise ValueError("No test images match the model's vocab.")

test_dl = learn.dls.test_dl(filtered_files)
preds, _ = learn.get_preds(dl=test_dl)
pred_labels = preds.argmax(dim=1)
true_labels = tensor([learn.dls.vocab.o2i[lbl] for lbl in filtered_labels])

accuracy = (pred_labels == true_labels).float().mean().item()
print(f'Accuracy on test images: {accuracy*100:.2f}%')