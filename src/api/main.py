import asyncio
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import uvicorn

from faststream.kafka.fastapi import KafkaRouter, Logger

router = KafkaRouter("kafka:9092", max_request_size=16000000)


responses_ml1 = dict()
responses_ml2 = dict()
responses_ml3 = dict()

def call():
    return True

@router.subscriber("from_ml1")
async def from_ml1(data, msg_id, d=Depends(call)):
    #print('from ml1', data, "msg_ID=", msg_id)
    if responses_ml1[msg_id] == 'timed_out':
        responses_ml1.pop(msg_id)
    else:
        responses_ml1[msg_id] = data

@router.subscriber("from_ml2")
async def from_ml2(data, msg_id,d=Depends(call)):
    #print('from mд2', data, msg_id)
    if responses_ml2[msg_id] == 'timed_out':
        responses_ml2.pop(msg_id)
    else:
        responses_ml2[msg_id] = data

@router.subscriber("from_ml3")
async def from_ml3(data, msg_id,d=Depends(call)):
    #print('from mд2', data, msg_id)
    if responses_ml3[msg_id] == 'timed_out':
        responses_ml3.pop(msg_id)
    else:
        responses_ml3[msg_id] = data

app = FastAPI(lifespan=router.lifespan_context)
app.include_router(router)

@app.get("/find")
async def root(data: dict | None = None):
    #data =  
    if data is None:
        data = {'models': ['Autoencoder', 'Isolation Forest', 'Prophet'], 'column_name': 'web_response'}
    #print('DATA:          ', data)
    #return data
    msg_id = max([len(responses_ml1), len(responses_ml2), len(responses_ml3)])
    responses_ml1[msg_id] = 'pending'
    responses_ml2[msg_id] = 'pending'
    responses_ml3[msg_id] = 'pending'
    if 'Autoencoder' in data['models']:
        await router.broker.publish({"msg_id": msg_id, 'column_name': data['column_name'], 'data_source': 'default'}, "to_ml1")
    if 'Isolation Forest' in data['models']:
        await router.broker.publish({"msg_id": msg_id, 'column_name': data['column_name'], 'data_source': 'default'}, "to_ml2")
    if 'Prophet' in data['models']:
        await router.broker.publish({"msg_id": msg_id, 'column_name': data['column_name'], 'data_source': 'default'}, "to_ml3")
    try:
        async with asyncio.timeout(120):
            while True:
                rdy = 0
                if ('Autoencoder' in data['models']) and (responses_ml1[msg_id] != 'pending'):
                    rdy += 1
                if ('Isolation Forest' in data['models']) and (responses_ml2[msg_id] != 'pending'):
                    rdy += 1
                if ('Prophet' in data['models']) and (responses_ml3[msg_id] != 'pending'):
                    rdy += 1
                if rdy == len(data['models']):
                    break
                
                await asyncio.sleep(0.25)
    except TimeoutError:
        print("The long operation timed out, but we've handled it.")
    if responses_ml1[msg_id] == 'pending':
        responses_ml1[msg_id] = 'timed_out'
    if responses_ml2[msg_id] == 'pending':
        responses_ml2[msg_id] = 'timed_out'
    if responses_ml3[msg_id] == 'pending':
        responses_ml3[msg_id] = 'timed_out'

    m1 = responses_ml1[msg_id]
    m2 = responses_ml2[msg_id]
    m3 = responses_ml2[msg_id]

    responses_ml1[msg_id] = 0
    responses_ml2[msg_id] = 0
    responses_ml3[msg_id] = 0

    return {'Autoencoder': m1, 'Isolation Forest': m2, 'Prophet': m3}#{"rmessage": [responses_ml1[msg_id], responses_ml2[msg_id]]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)