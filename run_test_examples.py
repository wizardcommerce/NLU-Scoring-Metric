from __future__ import annotations

import argparse
import os
import uuid

import numpy as np
import pandas as pd
from google.cloud import dialogflow_v2
from google.cloud.dialogflow_v2.types import Context
from google.cloud.dialogflow_v2.types import EntityType
from google.cloud.dialogflow_v2.types import EntityTypeBatch
from google.cloud.dialogflow_v2.types import EventInput
from google.cloud.dialogflow_v2.types import Intent
from google.cloud.dialogflow_v2.types import IntentBatch
from google.cloud.dialogflow_v2.types import QueryInput
from google.cloud.dialogflow_v2.types import QueryParameters
from google.cloud.dialogflow_v2.types import TextInput
from google.cloud.dialogflow_v2.types import Version
from google.protobuf.json_format import MessageToDict


# intents, upon detection, will transfer the ensuing convo to CX agents
# set invidual intents to False if you want to continue to send the convo to the NLU
TRANSFER_INTENTS = {
    'transfer to agent': True,
    'ask for recommendation': True,
    'buy product': True,
    'exit': True,  # this is also being keyword-matched at gate-keeper
    'get order status': True,
    'get discount': True,
    'get price': True,
    'get product availability': True,
    'get shipping availability': True,
    'get shipping estimate': True,
    'update address': True,
}

TEST_CHAR_LIMIT = 256

####################################################################################################
# Session info and getting and intent classification
####################################################################################################


def get_session_string(project_id: str, campaign_id: int, user_id: int):
    session_id = f'campaign-{campaign_id}-{user_id}'
    return f'projects/{project_id}/agent/sessions/{session_id}'


def get_initial_context(session_str, campaign_id):
    # set product list
    input_context = {
        'name': session_str + f'/contexts/{campaign_id}-followup',
        'lifespan_count': 2,
        'parameters': {},
    }
    session_vars = {
        'name': session_str + '/contexts/session-vars',
        'lifespan_count': 50,
        'parameters': {
            'products': [],  # TO BE SET - type: list[dict[str, str | int]]  list of products
        },
    }

    return [Context(input_context), Context(session_vars)]


def detect_intent(session_str: str, test_phrase: str, contexts: list[Context] or None):
    """
    detect an intent with contexts
    """
    # Create a client
    client = dialogflow_v2.SessionsClient()

    # Initialize request argument(s)
    request = dialogflow_v2.DetectIntentRequest(
        session=session_str,
        query_params=None if contexts is None else QueryParameters(
            contexts=contexts),
        query_input=QueryInput(text=TextInput(
            text=test_phrase[:TEST_CHAR_LIMIT], language_code='en-US')),
    )

    # Make the request
    response = client.detect_intent(request=request)
    return response


def delete_contexts(session_str):
    """
    delete all active contexts for the particular session associated with session_str
    """
    # Create a client
    client = dialogflow_v2.ContextsClient()

    # Initialize request argument(s)
    request = dialogflow_v2.DeleteAllContextsRequest(
        parent=session_str,
    )

    # Make the request
    client.delete_all_contexts(request=request)


####################################################################################################
# Run through a conversation and return bot data
####################################################################################################
def test_convo(queries: list[str], session_str: str, contexts: list[dict] or None):
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
    response = detect_intent(session_str, queries[0], contexts)
    log.append(response)
    columns.append([response.query_result.intent.display_name,
                   response.query_result.intent_detection_confidence])
    for _id, msg in enumerate(queries[1:]):
        # update contexts to mimic manipulation of contexts in the webhook
        # only under one circumstance we erase the previous contexts, that is, at create_order
        if response.query_result.action.endswith('create_order'):
            contexts = sorted(response.query_result.output_contexts,
                              key=lambda x: x.lifespan_count, reverse=True)[:2]
        else:
            contexts = response.query_result.output_contexts
        response = detect_intent(session_str, msg, contexts=contexts)
        log.append(response)
        columns.append([response.query_result.intent.display_name,
                       response.query_result.intent_detection_confidence])
        if 'is_fallback' in response.query_result.intent and response.query_result.intent.is_fallback\
                or TRANSFER_INTENTS.get(response.query_result.intent.display_name, False)\
                or response.query_result.intent.display_name.endswith('no'):
            break
    # this is particuarly important when you use the same session_str to test multiple convos
    delete_contexts(session_str)
    return columns, log


####################################################################################################
# Script arguments and main()
####################################################################################################
def main(args):
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        raise Exception(
            'Please set the file path to the DF API key creds json file: export GOOGLE_APPLICATION_CREDENTIALS=path')

    conv_n = args.conversation_delimiting_column
    text_column = args.text_column

    df = pd.read_csv(args.input)
    df[['pred_intent', 'confidence']] = None, None
    for conversation in df[conv_n].unique():
        # assuming that the messages have been ordered
        subdf = df.loc[df[conv_n] == conversation]

        session_str = get_session_string(
            project_id=args.project_id,
            campaign_id=args.campaign_id,
            user_id=args.user_id,
        )

        contexts = get_initial_context(
            session_str=session_str,
            campaign_id=args.campaign_id,
        )

        responses, _ = test_convo(
            queries=subdf[text_column].values,
            session_str=session_str,
            contexts=contexts,
        )

        responses = np.array(responses)
        idxs = subdf.index.values[:len(responses)]
        df['pred_intent'].loc[idxs], df['confidence'].loc[idxs] = responses[:,0], responses[:,1]

    # save to a file?
    df.to_csv(args.input, index=False)
    print(args.input, conv_n, text_column)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--text_column', type=str,
                        required=False, default='body')
    parser.add_argument('--conversation_delimiting_column',
                        required=False, default='conversation_id')
    parser.add_argument('--campaign_id', required=False, type=int, help='One of set: [46, 47, 44, 45]. See https://www.mindomo.com/mindmap/draft-3d2d6c05bea043c590268db245eddbc7 for the flow the number is associated with')
    parser.add_argument('--user_id', required=False, type=int, default=1)
    parser.add_argument('--project_id', type=str,
                        required=True, help='the unique DF project id')

    args = parser.parse_args()
    main(args=args)
