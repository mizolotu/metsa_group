import datetime
import logging
import azure.functions as func

from .utils import get_data_rows, predict, update_row

null_column = 'windspeed'
nfeatures = 3

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)

    df = get_data_rows(null_column)
    vals = df.values
    if len(vals) > 0 and len(vals.shape) == 2:
        ids = vals[:, 0]
        x = vals[:, 1:nfeatures+1]
        wss = predict(x)
        for id, ws in zip(ids, wss):
            update_row(id, ws)
            logging.info(f'row {id} has been updated with ws value = {ws}')