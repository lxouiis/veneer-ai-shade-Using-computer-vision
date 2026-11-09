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
        "headers": {
            "Content-Type": "application/json",
            # start_page.html calls this from wherever it's hosted (S3) -
            # needs CORS open since it's a different origin than this Lambda.
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
