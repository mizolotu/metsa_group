import azure.functions as func

from .utils import load_columns, insert_data_row

def main(req: func.HttpRequest) -> func.HttpResponse:

    columns = load_columns('metainfo.json')
    param_names = columns['features'] + [columns['label']] + [columns['timestamp']]

    sample = {}
    for name in param_names:
        value = req.params.get(name)
        if not value:
            try:
                req_body = req.get_json()
            except ValueError:
                pass
            else:
                value = req_body.get(name)
        if value is not None:
            sample[name] = value

    if len(sample) > 0:
        insert_data_row(sample)
        return func.HttpResponse(f"Data row has been inserted!")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully, but you have to pass some data.",
             status_code=200
        )