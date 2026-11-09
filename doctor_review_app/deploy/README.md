# Deploying to AWS EC2

One small always-on instance, running the Streamlit app as a background
service, with HTTPS via Caddy so it opens cleanly on Doc Sans's iPad (real
padlock, not a security warning). With your $200 new-account signup credits this is
effectively free for the life of this project - a t3.medium is roughly
$30/month on-demand, and you can stop the instance between review sessions
to pay only for the hours it's actually running.

## New to AWS/Linux servers? Watch these first (in order)
1. **Launch an Ubuntu EC2 instance**: https://www.youtube.com/watch?v=Xyv5rq5gX50
2. **Deploy a Streamlit app on EC2** (the closest match to this exact setup):
   https://www.youtube.com/watch?v=DflWqmppOAg
   - A second walkthrough if the first doesn't click: https://www.youtube.com/watch?v=oynd7Xv2i9Y
3. **Get a free domain name with DuckDNS** (needed for step 2 below, so Caddy
   can issue a real HTTPS certificate): https://www.youtube.com/watch?v=2NB1c63kKO0

None of these match this app exactly (yours needs systemd + Caddy on top,
covered in the steps below) - watch them for the general shape of "launch an
instance, SSH in, run a Python web app," then follow this README for the
specifics of this project.

## 1. Launch the instance (AWS Console -> EC2 -> Launch Instance)
- AMI: **Ubuntu 22.04 LTS**
- Instance type: **t3.medium** (4GB RAM, 2 vCPU) - MedSAM's ViT-B encoder +
  YOLO loaded together need more than the 1GB a smaller/free-tier instance
  gives you; t3.medium is the safe choice, not t2.micro/t3.small.
- Storage: 20GB gp3 (default 8GB is tight once torch + both checkpoints are
  on disk)
- Security group: allow **22 (SSH)** from your IP only, **80 and 443
  (HTTP/HTTPS)** from anywhere (0.0.0.0/0) - Caddy needs 80 briefly to prove
  domain ownership for the TLS certificate, then serves on 443.
- Key pair: create/download one for SSH access.
- After launch: **Elastic IP** -> allocate one and associate it with this
  instance, so the public address doesn't change if it ever restarts (a free
  DNS name below points at this fixed IP).

## 2. Get a free domain name pointed at the Elastic IP
Caddy needs a real domain to issue a TLS certificate (plain IP addresses
can't get one). Easiest free option: **duckdns.org** - sign in, create a
subdomain (e.g. `toothreview.duckdns.org`), point it at your instance's
Elastic IP. Takes under 2 minutes, no cost.

## 3. SSH in and create a dedicated user
```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
sudo adduser --disabled-password --gecos "" toothreview
sudo mkdir -p /home/toothreview/app
sudo chown -R toothreview:toothreview /home/toothreview/app
```

## 4. Install dependencies and copy the app
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
sudo -u toothreview -i
cd ~/app
git clone <your-repo-url> .       # or scp the doctor_review_app/ folder up
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
# copy medsam_tooth_best.pth and tooth_yolo_best.pt into models/ here
# (scp them from your machine, or download from wherever you stored them)
exit   # back to your ubuntu user
```

## 5. Fill in the secrets file
```bash
sudo -u toothreview tee /home/toothreview/app/.env > /dev/null <<'EOF'
GDRIVE_CLIENT_ID=...
GDRIVE_CLIENT_SECRET=...
GDRIVE_REFRESH_TOKEN=...
GDRIVE_ROOT_ID=...
MEDSAM_CHECKPOINT_PATH=/home/toothreview/app/models/medsam_tooth_best.pth
YOLO_CHECKPOINT_PATH=/home/toothreview/app/models/tooth_yolo_best.pt
EOF
sudo chmod 600 /home/toothreview/app/.env
```
(Values come from `setup_drive.py`'s printed output - see module_7's own
`notes.md`.)

## 6. Install the systemd service
```bash
sudo cp /home/toothreview/app/deploy/tooth-review.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tooth-review
sudo systemctl status tooth-review   # should show "active (running)"
```

## 7. Install Caddy and point it at the app
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy

# Edit the domain in deploy/Caddyfile to your actual duckdns subdomain first
sudo cp /home/toothreview/app/deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

## 8. Done
Open `https://toothreview.duckdns.org` (your actual subdomain) from any
browser, including Doc Sans's iPad Safari. Caddy auto-renews the TLS
certificate - nothing more to maintain.

## 9. The "one-tap start" button for Doc Sans (Lambda + S3)
The instance should stay **off** between review sessions to save credits -
but if it's off, there's nothing running to show her a toggle. The fix is a
tiny piece that's always available regardless of instance state: an AWS
Lambda function with its own permanent URL, plus a small static page she
actually bookmarks (not the Streamlit URL itself).

**9a. Deploy the Lambda** (console: Lambda -> Create function)
- Runtime: Python 3.12. Paste in `lambda_starter/start_stop_ec2.py`.
- Environment variable: `INSTANCE_ID` = your EC2 instance's ID.
- Execution role: attach a policy based on `lambda_starter/iam_policy.json`
  (fill in your region/account/instance ID) - scoped to exactly this one
  instance for Start/Stop, with `DescribeInstances` on `*` since AWS doesn't
  support scoping that particular action to one resource.
- Configuration -> Function URL -> Create. Auth type: NONE (this only
  starts/stops one specific instance, not a sensitive action - but if you'd
  rather lock it down, IAM auth is the stricter option). Copy the resulting
  URL.
- Add it to the instance's `.env` so the app's sidebar shows a "Stop
  Instance" button (lets Doc Sans end her own session, not just start it):
  ```bash
  echo "LAMBDA_STARTER_URL=<the function url>" | sudo -u toothreview tee -a /home/toothreview/app/.env
  sudo systemctl restart tooth-review
  ```

**9b. Host the start page** (console: S3 -> Create bucket)
- Create a bucket, enable **Static website hosting** in its properties.
- In `lambda_starter/start_page.html`, replace `LAMBDA_URL` with the
  Function URL from 9a, and `APP_URL` with your DuckDNS domain
  (`https://toothreview.duckdns.org`).
- Upload the edited file, make it public (bucket policy allowing
  `s3:GetObject`), and use the bucket's static website endpoint as the link.

**9c. Give Doc Sans the S3 link, not the DuckDNS one.** She bookmarks the S3
page on her iPad. Tapping **Start** calls the Lambda, waits for the instance
to boot (usually under a minute), then auto-redirects to the real app once
it's actually reachable - she never needs to know an EC2 instance is
involved at all.

## Updating the app later
```bash
sudo -u toothreview -i
cd ~/app && git pull            # or re-copy changed files
source .venv/bin/activate && pip install -r requirements.txt  # if deps changed
exit
sudo systemctl restart tooth-review
```

## Pausing to save credits
The Lambda from section 9 is how Doc Sans starts/stops it day-to-day. For
you directly, the same thing works from a terminal:
```bash
# From your own machine, AWS CLI, or the console:
aws ec2 stop-instances --instance-ids <your-instance-id>
# Elastic IP stays reserved (small idle cost) unless you release it, so the
# domain keeps working once you start the instance again:
aws ec2 start-instances --instance-ids <your-instance-id>
```
Worth adding a scheduled nightly auto-stop (EventBridge -> this same Lambda
with `action=stop`) too, as a safety net in case anyone forgets to stop it
manually.
