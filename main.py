from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt

data = []

for record in SeqIO.parse(
    "data/sequences.fasta",
    "fasta"
):

    seq = str(record.seq)

    length = len(seq)

    gc = (
        seq.count("G")
        + seq.count("C")
    ) / length

    a = seq.count("A") / length
    t = seq.count("T") / length
    g = seq.count("G") / length
    c = seq.count("C") / length

    data.append([
        record.id,
        length,
        gc,
        a,
        t,
        g,
        c
    ])

df = pd.DataFrame(
    data,
    columns=[
        "ID",
        "Length",
        "GC_Content",
        "A",
        "T",
        "G",
        "C"
    ]
)

df["Label"] = [
    1,
    0,
    1,
    1,
    0
]

print(df)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X = df.drop(
    ["ID", "Label"],
    axis=1
)

y = df["Label"]

print("\nDataset Summary")
print("----------------")
print("Number of Sequences:", len(df))
print("Number of Features:", X.shape[1]) 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=1
)

knn.fit(
    X_train,
    y_train
)

predictions = knn.predict(
    X_test
)

knn_accuracy = accuracy_score(y_test, predictions)


print("\nKNN Accuracy:", knn_accuracy)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

nb_accuracy = accuracy_score(
    y_test,
    nb_pred
)

print("Naive Bayes Accuracy:", nb_accuracy)
print("\nModel Comparison")
print("----------------")
print("KNN Accuracy:", knn_accuracy)
print("Naive Bayes Accuracy:", nb_accuracy)
results = {
    "KNN": knn_accuracy,
    "Naive Bayes": nb_accuracy
}

best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", results[best_model])

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

models = [
    "KNN",
    "Naive Bayes"
]

accuracies = [
    knn_accuracy,
    nb_accuracy
]

plt.bar(models, accuracies)

plt.title("DNA Sequence Classification")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.savefig("dna_model_comparison.png")

plt.show()