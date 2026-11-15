"""AWS Lambda handler - this is the one piece that stays available even when
the EC2 instance is OFF, so Doc Sans has something to actually tap to turn
it back on. Deploy with a Lambda Function URL (not API Gateway - simpler,
and this doesn't need anything API Gateway adds).

Environment variable required: INSTANCE_ID (the EC2 instance to control).

Query param `action`:
  - start  -> starts the instance
  - stop   -> stops the instance
  - status -> (default) just reports current state, no action taken
"""
import json
import os
import boto3

INSTANCE_ID = os.environ["INSTANCE_ID"]
ec2 = boto3.client("ec2")


def handler(event, context):
    params = event.get("queryStringParameters") or {}
    action = params.get("action", "status")

    if action == "start":
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
    elif action == "stop":
        ec2.stop_instances(InstanceIds=[INSTANCE_ID])
    elif action != "status":
        return _response(400, {"error": "action must be 'start', 'stop', or 'status'"})

    state = ec2.describe_instances(InstanceIds=[INSTANCE_ID])["Reservations"][0]["Instances"][0]["State"]["Name"]
    return _response(200, {"state": state})


def _response(status_code, body):
    return {
        "statusCode": status_code,
        # CORS is handled by the Function URL's own CORS config (set via
        # `aws lambda create-function-url-config --cors ...`), not here -
        # adding a second "Access-Control-Allow-Origin" header on top of
        # that produces TWO such headers in the same response, which
        # browsers reject outright as an invalid CORS response (the fetch()
        # call in start_page.html fails client-side even though the request
        # genuinely succeeded server-side - confirmed via curl showing a
        # clean 200 with duplicate headers before this fix).
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
