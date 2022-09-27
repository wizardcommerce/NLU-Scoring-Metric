import argparse
import numpy as np
import pandas as pd
import os
import uuid
from google.cloud import dialogflow_v2
from google.cloud.dialogflow_v2.types import (Intent, IntentBatch, QueryInput, TextInput,
EntityType, EntityTypeBatch, Version, QueryParameters, Context, EventInput)
from google.protobuf.json_format import MessageToDict

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  "/Users/zacharyrosen/workspace/proj/NLUscoring/CB58/CB60/credentials/campaign-prototype-oxtt-ff3908a816e1.json"
def tap(TEST: int):
    print(TEST)


####################################################################################################
### Session info and getting and intent classification
####################################################################################################
def get_session_string(campaign_id: str, user_id: str):
    session_id = f'campaign-{campaign_id}-{user_id}'
    return f"projects/campaign-prototype-oxtt/agent/sessions/{session_id}"

def get_initial_context(session_str,campaign_id):
    # set product list
    input_context = {
        'name': session_str + f'/contexts/{campaign_id}-followup',
        'lifespan_count': 2,
        'parameters': {}
    }
    session_vars = {
        'name': session_str + '/contexts/session-vars',
        'lifespan_count': 50,
        'parameters': {
            'products': []  # TO BE SET - type: list[dict[str, str | int]]  list of products
        }
    }

    return [Context(input_context), Context(session_vars)]

def detect_intent(session_str: str, test_phrase: str, contexts: list[Context] or None, TEST_CHAR_LIMIT: int or None):
    """
    detect an intent with contexts
    """
    # Create a client
    client = dialogflow_v2.SessionsClient()

    # Initialize request argument(s)
    request = dialogflow_v2.DetectIntentRequest(
        session=session_str,
        query_params=None if contexts is None else QueryParameters(contexts=contexts),
        query_input=QueryInput(text=TextInput(text=test_phrase[:TEST_CHAR_LIMIT], language_code='en-US'))
    )

    # Make the request
    response = client.detect_intent(request=request)
    return response



####################################################################################################
### Run through a conversation and return bot data
####################################################################################################
def test_convo(queries: list[str], session_str: str, contexts: list[dict] or None, TEST_CHAR_LIMIT: int, TRANSFER_INTENTS: list[str]):
    """
    test one convo with a list of customer inputs in chrono sequence
    Args:
        queries: sequence of customer input
        session_str: the DF agent query str
    Returns:
        a list of DF responses
    """
    log = []
    columns = []

    # the first customer input
    response = detect_intent(session_str, queries[0], contexts, TEST_CHAR_LIMIT=TEST_CHAR_LIMIT)
    log.append(response)
    columns.append([response.query_result.intent.display_name, response.query_result.intent_detection_confidence])

    for _id, msg in enumerate(queries[1:]):
        response = detect_intent(session_str, msg, contexts=response.query_result.output_contexts)
        log.append(response)
        columns.append([response.query_result.intent.display_name, response.query_result.intent_detection_confidence])
        if "is_fallback" in response.query_result.intent and response.query_result.intent.is_fallback\
            or response.query_result.intent.display_name in TRANSFER_INTENTS\
            or response.query_result.intent.display_name.endswith('no'):
            break

    return columns,log



####################################################################################################
### Script arguments and main()
####################################################################################################
def main(args):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/zacharyrosen/workspace/proj/NLUscoring/CB58/CB60/credentials/campaign-prototype-oxtt-ff3908a816e1.json"

    conv_n = args.conversation_delimiting_column
    text_column = args.text_column
    TEST_CHAR_LIMIT = args.CHAR_LIMIT

    TRANSFER_INTENTS = [
        'transfer to agent', 'ask for recommendation',
        'buy product', 'exit',
        'get order status'
    ]

    # df = pd.read_csv(args.input)
    # df[['pred_intent','confidence']] = None, None
    # for conversation in df[conv_n].unique():
    #
    #     subdf = df.loc[df[conv_n].isin([conversation])]
    #
    #     session_str = get_session_string(
    #         campaign_id=args.campaign_id,
    #         user_id=args.user_id
    #     )
    #
    #     contexts = get_initial_context(
    #         session_str=session_str,
    #         campaign_id=args.campaign_id
    #     )
    #
    #     responses, _ = test_convo(
    #         queries=subdf[text_column].values,
    #         session_str=session_str,
    #         contexts=contexts,
    #         TEST_CHAR_LIMIT=TEST_CHAR_LIMIT,
    #         TRANSFER_INTENTS=TRANSFER_INTENTS
    #     )
    #
    #     idxs = subdf.index.values[:len(responses)]
    #     df[['pred_intent','confidence']].loc[idxs] = np.array(responses)


    print(args.input,conv_n,text_column, TEST_CHAR_LIMIT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,required=True,)
    parser.add_argument('--CHAR_LIMIT',type=int,required=False,default=None)
    parser.add_argument('--text_column', type=str,required=False,default='body')
    parser.add_argument('--conversation_delimiting_column',required=False,default='conversation_id')
    parser.add_argument('--campaign_id', required=False, default='')
    parser.add_argument('--user_id',required=False,default='')

    args = parser.parse_args()
    main(args=args)

