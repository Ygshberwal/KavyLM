import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
 


data=pd.read_csv("Dataset_Poems.csv")

label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["Type"])

data=data.iloc[:,[1,12]]
data.columns=["text", "labels"]

train_data, eval_data = train_test_split(data, test_size=0.2, random_state=43)



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_args = ClassificationArgs(
    num_train_epochs=10,
    train_batch_size=8,
    eval_batch_size=8,
    overwrite_output_dir=True,  # Clear previous outputs
    save_eval_checkpoints=False,
    save_model_every_epoch=False
)


accuracies = []
conf_matrices = []
fold = 1

for train_index, test_index in kf.split(data["text"], data["labels"]):
    print(f"\nFold {fold}:")

    # Train-test split

    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=43, stratify=data["labels"])


    # Create model
    model = ClassificationModel(
        "xlmroberta",
        "xlm-roberta-base",
        num_labels=len(data["labels"].unique()),
        args=model_args,
        use_cuda=True
    )

    # Train model
    model.train_model(train_data)

    # Evaluate on test data
    result, model_outputs, _ = model.eval_model(eval_data)
    predictions = model_outputs.argmax(axis=1)
    labels = eval_data["labels"].values

    # Calculate accuracy
    acc = accuracy_score(labels, predictions)
    accuracies.append(acc)
    print(f"Accuracy for fold {fold}: {acc * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    conf_matrices.append(cm)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.show()

    fold += 1

# Cross-validation summary
print("\nCross-Validation Summary:")
print(accuracies)
print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(classification_report(eval_data["labels"], predictions, target_names=["Doha", "Kavita", "Nazm_Ghazal", "Sher/Shayari"]))
