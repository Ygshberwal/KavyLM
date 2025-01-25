import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import accuracy_score



data=pd.read_csv("Dataset_Poems.csv")

label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["Type"])

data=data.iloc[:,[1,12]]

train_data, eval_data = train_test_split(data, test_size=0.2, random_state=43)

train_data.columns=["text", "labels"]
eval_data.columns=["text", "labels"]


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = ClassificationArgs(
    num_train_epochs=6,
    train_batch_size=8,
    eval_batch_size=8
)
# Create a ClassificationModel
model = ClassificationModel(
    "xlmroberta", "xlm-roberta-base", num_labels=4, args=model_args
)

# Train the model
model.train_model(train_data)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_data)



# Get the predictions and labels
predictions = model_outputs.argmax(axis=1)  # Get the class with the highest score
labels = eval_data['labels'].values  # True labels

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

