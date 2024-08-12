
def calculate_determination_coefficient(y_true, y_pred):
    """
    Calculate the determination coefficient of the model.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: determination coefficient
    """
    y_true_mean = sum(y_true) / len(y_true)
    ss_total = sum((y - y_true_mean) ** 2 for y in y_true)
    ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_true, y_pred))
    return 1 - ss_res / ss_total
