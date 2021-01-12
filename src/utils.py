import numpy as np
from sklearn.metrics import confusion_matrix


def flat_accuracy(valid_tags, pred_tags):

    """
    Define a flat accuracy metric to use while training the model.
    """

    return (np.array(valid_tags) == np.array(pred_tags)).mean()

def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content
