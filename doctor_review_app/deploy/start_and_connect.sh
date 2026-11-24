#!/bin/bash
# Starts the EC2 instance (if stopped), fixes the SSH security group rule to
# allow whatever IP you're on right now (looked up automatically - nothing
# to edit here between runs), then SSHes in. Run with:
#   bash deploy/start_and_connect.sh
set -euo pipefail

INSTANCE_ID="i-0f5a8e6709f42f178"
SECURITY_GROUP_ID="sg-0b38709332311736f"
SSH_KEY="$HOME/Downloads/tooth-review-key.pem"
AWS_PROFILE="veneer-shade"
AWS_REGION="ap-southeast-2"

aws_() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

echo "==> Checking instance state..."
STATE=$(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "$STATE" != "running" ]; then
    echo "==> Instance is '$STATE' - starting it..."
    aws_ ec2 start-instances --instance-ids "$INSTANCE_ID" > /dev/null
    echo "==> Waiting for it to reach 'running' (can take ~30-60s)..."
    aws_ ec2 wait instance-running --instance-ids "$INSTANCE_ID"
else
    echo "==> Already running."
fi

PUBLIC_IP=$(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "==> Instance public IP: $PUBLIC_IP"

echo "==> Looking up your current public IP..."
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "==> Your current IP: $MY_IP"

echo "==> Checking what's currently authorized for SSH (port 22)..."
# All CIDRs currently allowed, one per line - there can be more than one if
# an earlier manual fix added a new IP without removing the old one (exactly
# what happened before this script existed).
OLD_IPS=$(aws_ ec2 describe-security-groups --group-ids "$SECURITY_GROUP_ID" \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`].IpRanges[].CidrIp" \
    --output text | tr '\t' '\n' | sed 's#/32##')

ALREADY_ALLOWED=false
while IFS= read -r ip; do
    [ "$ip" = "$MY_IP" ] && ALREADY_ALLOWED=true
done <<< "$OLD_IPS"

if [ "$ALREADY_ALLOWED" = true ]; then
    echo "==> Your current IP is already allowed."
else
    echo "==> Authorizing current IP ($MY_IP)..."
    aws_ ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP_ID" \
        --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":22,\"ToPort\":22,\"IpRanges\":[{\"CidrIp\":\"$MY_IP/32\",\"Description\":\"my IP only\"}]}]" \
        > /dev/null
fi

# Clean up every OTHER stale IP (anything that isn't the current one) so
# the rule list doesn't just keep growing forever.
while IFS= read -r ip; do
    if [ -n "$ip" ] && [ "$ip" != "None" ] && [ "$ip" != "$MY_IP" ]; then
        echo "==> Revoking stale IP ($ip)..."
        aws_ ec2 revoke-security-group-ingress --group-id "$SECURITY_GROUP_ID" \
            --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":22,\"ToPort\":22,\"IpRanges\":[{\"CidrIp\":\"$ip/32\"}]}]" \
            > /dev/null
    fi
done <<< "$OLD_IPS"

echo "==> Connecting..."
ssh -i "$SSH_KEY" "ubuntu@$PUBLIC_IP"
