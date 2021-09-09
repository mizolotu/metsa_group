import logging
import azure.functions as func

from .utils import select_last_prediction_row, select_last_data_rows, select_new_data_samples, prepare_samples, predict, insert_prediction_row

def main(req: func.HttpRequest) -> func.HttpResponse:

    last_prediction_rows = select_last_prediction_row()
    last_data_rows = select_last_data_rows()

    if len(last_data_rows) > 0:
        if len(last_prediction_rows) == 0:
            rows_to_predict = [last_data_rows[0]]
        else:
            rows_to_predict = select_new_data_samples(last_data_rows, last_prediction_rows[0])

    for row in rows_to_predict:

        # predictions for each class

        timestamp, samples, label = prepare_samples(row)
        predictions, errors = [], []

        logging.info(f'Real: {label}')

        for sample in samples:
            prediction, model = predict(sample)
            predictions.append(prediction)
            if prediction is not None and label is not None:
                error = abs(prediction - label)
            else:
                error = None
            errors.append(error)

            logging.info(f'Prediction: {prediction}, model: {model}, error: {error}')

        # insert the result into table

        insert_prediction_row(timestamp, label, predictions, errors)

    return func.HttpResponse(f"{len(rows_to_predict)} rows have been added to the prediction table!")